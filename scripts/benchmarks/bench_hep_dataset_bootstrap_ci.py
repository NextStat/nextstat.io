#!/usr/bin/env python3
"""HEP dataset-level bootstrap CI calibration (percentile vs BCa).

This benchmark targets *dataset-level* calibration:
- each outer run regenerates a fresh observed dataset (ROOT TTree),
- observed POI is fitted with `nextstat unbinned-fit`,
- bootstrap replicates are generated with `nextstat unbinned-fit-toys --gen mle`,
- percentile and BCa intervals for POI are computed from bootstrap `poi_hat`,
- coverage is evaluated against known generation truth.

BCa acceleration uses leave-one-out jackknife estimates of the sample mean.
This is exact for the Gaussian-mean setup used here (fixed sigma).

Observed ROOT writing supports two modes:
- `uproot` (rootless, preferred in `--root-writer auto`)
- `root` CLI macro writer fallback.
"""

from __future__ import annotations

import argparse
import json
import math
import shutil
import statistics
import subprocess
import tempfile
import time
from pathlib import Path
from statistics import NormalDist
from typing import Any

import numpy as np

from _parse_utils import parse_json_stdout


NORM = NormalDist()


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


def _quantile(sorted_values: list[float], p: float) -> float:
    if not sorted_values:
        raise ValueError("quantile requires non-empty input")
    if p <= 0.0:
        return sorted_values[0]
    if p >= 1.0:
        return sorted_values[-1]
    if len(sorted_values) == 1:
        return sorted_values[0]
    pos = p * (len(sorted_values) - 1)
    lo = int(math.floor(pos))
    hi = int(math.ceil(pos))
    if lo == hi:
        return sorted_values[lo]
    w = pos - lo
    return sorted_values[lo] * (1.0 - w) + sorted_values[hi] * w


def _percentile_interval(values: list[float], conf_level: float) -> tuple[float, float]:
    alpha_low = (1.0 - conf_level) / 2.0
    alpha_high = 1.0 - alpha_low
    s = sorted(values)
    return _quantile(s, alpha_low), _quantile(s, alpha_high)


def _jackknife_acceleration_from_mean(values: np.ndarray) -> float | None:
    n = int(values.size)
    if n < 3:
        return None
    total = float(values.sum())
    jack = (total - values) / float(n - 1)
    tbar = float(jack.mean())
    diffs = tbar - jack
    num = float(np.sum(diffs**3))
    den = float(np.sum(diffs**2))
    if den <= 0.0:
        return 0.0
    den = 6.0 * (den ** 1.5)
    if not math.isfinite(den) or den == 0.0:
        return None
    a = num / den
    if not math.isfinite(a):
        return None
    return float(a)


def _bca_interval(
    bootstrap_estimates: list[float],
    observed_estimate: float,
    conf_level: float,
    acceleration: float | None,
) -> tuple[float, float, dict[str, Any]]:
    alpha_low = (1.0 - conf_level) / 2.0
    alpha_high = 1.0 - alpha_low
    s = sorted(bootstrap_estimates)
    n = len(s)
    if n < 2:
        lo, hi = _percentile_interval(s, conf_level)
        return lo, hi, {
            "requested_method": "bca",
            "effective_method": "percentile",
            "fallback_reason": "need at least 2 bootstrap estimates",
            "z0": None,
            "acceleration": acceleration,
            "alpha_low": alpha_low,
            "alpha_high": alpha_high,
            "alpha_low_adj": None,
            "alpha_high_adj": None,
            "n_bootstrap": n,
        }

    eps = 1.0 / (2.0 * n)
    prop = sum(1 for x in s if x < observed_estimate) / n
    prop = min(max(prop, eps), 1.0 - eps)
    z0 = float(NORM.inv_cdf(prop))

    if acceleration is None or not math.isfinite(acceleration):
        lo, hi = _percentile_interval(s, conf_level)
        return lo, hi, {
            "requested_method": "bca",
            "effective_method": "percentile",
            "fallback_reason": "invalid acceleration estimate",
            "z0": z0,
            "acceleration": acceleration,
            "alpha_low": alpha_low,
            "alpha_high": alpha_high,
            "alpha_low_adj": None,
            "alpha_high_adj": None,
            "n_bootstrap": n,
        }

    def _adjust(alpha: float) -> float | None:
        z = float(NORM.inv_cdf(alpha))
        denom = 1.0 - acceleration * (z0 + z)
        if not math.isfinite(denom) or abs(denom) < 1e-12:
            return None
        z_adj = z0 + (z0 + z) / denom
        if not math.isfinite(z_adj):
            return None
        p = float(NORM.cdf(z_adj))
        if not math.isfinite(p):
            return None
        return min(max(p, 0.0), 1.0)

    low_adj = _adjust(alpha_low)
    high_adj = _adjust(alpha_high)
    if low_adj is None or high_adj is None or low_adj >= high_adj:
        lo, hi = _percentile_interval(s, conf_level)
        return lo, hi, {
            "requested_method": "bca",
            "effective_method": "percentile",
            "fallback_reason": "invalid adjusted alpha",
            "z0": z0,
            "acceleration": acceleration,
            "alpha_low": alpha_low,
            "alpha_high": alpha_high,
            "alpha_low_adj": low_adj,
            "alpha_high_adj": high_adj,
            "n_bootstrap": n,
        }

    lo = _quantile(s, low_adj)
    hi = _quantile(s, high_adj)
    return lo, hi, {
        "requested_method": "bca",
        "effective_method": "bca",
        "fallback_reason": None,
        "z0": z0,
        "acceleration": acceleration,
        "alpha_low": alpha_low,
        "alpha_high": alpha_high,
        "alpha_low_adj": low_adj,
        "alpha_high_adj": high_adj,
        "n_bootstrap": n,
    }


def _write_root_writer_macro(path: Path) -> None:
    src = r'''
#include <TFile.h>
#include <TTree.h>
#include <fstream>

void write_tree_from_txt(const char* txt_path, const char* out_path) {
  std::ifstream in(txt_path);
  if (!in.good()) {
    throw std::runtime_error("failed to open txt input");
  }

  TFile f(out_path, "RECREATE");
  TTree t("events", "events");
  double mbb = 0.0;
  t.Branch("mbb", &mbb, "mbb/D");

  while (in >> mbb) {
    t.Fill();
  }
  t.Write();
  f.Close();
}
'''
    path.write_text(src, encoding="utf-8")


def _write_values_txt(path: Path, values: np.ndarray) -> None:
    with path.open("w", encoding="utf-8") as f:
        for v in values:
            f.write(f"{float(v):.17g}\n")


def _write_spec(
    path: Path,
    *,
    data_file: Path,
    poi_name: str,
    poi_init: float,
    poi_bounds: tuple[float, float],
    sigma: float,
    n_events: int,
) -> None:
    spec = {
        "schema_version": "nextstat_unbinned_spec_v0",
        "model": {
            "poi": poi_name,
            "parameters": [
                {"name": poi_name, "init": poi_init, "bounds": [poi_bounds[0], poi_bounds[1]]},
                {"name": "gauss_sigma", "init": sigma, "bounds": [sigma, sigma]},
            ],
        },
        "channels": [
            {
                "name": "SR",
                "include_in_fit": True,
                "data": {"file": str(data_file), "tree": "events"},
                "observables": [{"name": "mbb", "bounds": [0.0, 500.0]}],
                "processes": [
                    {
                        "name": "p",
                        "pdf": {
                            "type": "gaussian",
                            "observable": "mbb",
                            "params": [poi_name, "gauss_sigma"],
                        },
                        "yield": {"type": "fixed", "value": float(n_events)},
                    }
                ],
            }
        ],
    }
    path.write_text(json.dumps(spec, indent=2), encoding="utf-8")


def _run_json(cmd: list[str]) -> tuple[dict[str, Any], float]:
    t0 = time.perf_counter()
    try:
        proc = subprocess.run(cmd, capture_output=True, text=True, check=True)
    except subprocess.CalledProcessError as exc:
        stderr = (exc.stderr or "").strip()
        stderr_short = stderr[:400] if stderr else ""
        raise RuntimeError(
            f"command failed (exit={exc.returncode}): {' '.join(cmd)}; stderr={stderr_short}"
        ) from exc
    wall_s = time.perf_counter() - t0
    return parse_json_stdout(proc.stdout), wall_s


def _run_root_write(macro_path: Path, txt_path: Path, out_root: Path) -> None:
    cmd = [
        "root",
        "-l",
        "-b",
        "-q",
        f'{macro_path}(\"{txt_path}\",\"{out_root}\")',
    ]
    subprocess.run(cmd, check=True, capture_output=True, text=True)


def _has_uproot() -> bool:
    try:
        import uproot  # noqa: F401
    except Exception:
        return False
    return True


def _resolve_root_writer(mode: str) -> str:
    if mode == "auto":
        if _has_uproot():
            return "uproot"
        if shutil.which("root") is not None:
            return "root-cli"
        raise RuntimeError(
            "no ROOT writer available: install `uproot` (rootless path) or provide `root` CLI"
        )
    if mode == "uproot":
        if not _has_uproot():
            raise RuntimeError("--root-writer uproot requested, but `uproot` is not importable")
        return "uproot"
    if mode == "root-cli":
        if shutil.which("root") is None:
            raise RuntimeError("--root-writer root-cli requested, but `root` binary is missing")
        return "root-cli"
    raise RuntimeError(f"unsupported --root-writer mode: {mode}")


def _write_root_with_uproot(out_root: Path, values: np.ndarray) -> None:
    try:
        import uproot
    except Exception as exc:
        raise RuntimeError(
            "failed to import `uproot` for rootless ROOT writing; install via `pip install uproot`"
        ) from exc
    with uproot.recreate(out_root) as f:
        # Use mktree to guarantee classic TTree (not RNTuple).
        # uproot >=5 may write RNTuple by default via __setitem__.
        f.mktree("events", {"mbb": np.asarray(values, dtype=np.float64)})


def _aggregate(rows: list[dict[str, Any]]) -> dict[str, Any]:
    widths = [float(r["width"]) for r in rows]
    walls = [float(r["wall_total_s"]) for r in rows]
    coverage = sum(1 for r in rows if r["contains_true"]) / len(rows)
    center_bias = [float(r["center_minus_true"]) for r in rows]
    t0_bias = [float(r["obs_hat_minus_true"]) for r in rows]
    fallback_count = sum(1 for r in rows if r.get("fallback_reason"))
    effective_bca_count = sum(1 for r in rows if r.get("effective_method") == "bca")
    return {
        "runs": len(rows),
        "coverage_vs_true_poi": coverage,
        "median_width": statistics.median(widths),
        "mean_width": statistics.fmean(widths),
        "median_wall_total_s": statistics.median(walls),
        "mean_wall_total_s": statistics.fmean(walls),
        "median_center_minus_true": statistics.median(center_bias),
        "mean_center_minus_true": statistics.fmean(center_bias),
        "median_obs_hat_minus_true": statistics.median(t0_bias),
        "mean_obs_hat_minus_true": statistics.fmean(t0_bias),
        "fallback_count": fallback_count,
        "effective_bca_count": effective_bca_count,
    }


def _write_markdown(summary: dict[str, Any], out_path: Path) -> None:
    lines = [
        "# HEP Dataset-Level Bootstrap CI Benchmark",
        "",
        "Generated by `scripts/benchmarks/bench_hep_dataset_bootstrap_ci.py`.",
        "",
    ]
    for scenario, methods in summary["scenarios"].items():
        lines.extend(
            [
                f"## Scenario: `{scenario}`",
                "",
                "| Method | Coverage vs true POI | Median width | Median wall total (s) | Median center minus true | Fallbacks |",
                "|---|---:|---:|---:|---:|---:|",
            ]
        )
        for method in ("percentile", "bca"):
            m = methods[method]
            lines.append(
                f"| {method} | {m['coverage_vs_true_poi']:.3f} | {m['median_width']:.6f} | "
                f"{m['median_wall_total_s']:.3f} | {m['median_center_minus_true']:+.6f} | "
                f"{m['fallback_count']} |"
            )
        lines.append("")
    out_path.write_text("\n".join(lines), encoding="utf-8")


def main() -> None:
    ap = argparse.ArgumentParser(description="HEP dataset-level bootstrap CI benchmark")
    ap.add_argument("--runs", type=int, default=8)
    ap.add_argument("--seed0", type=int, default=100)
    ap.add_argument("--n-events", type=int, default=400)
    ap.add_argument("--sigma-true", type=float, default=30.0)
    ap.add_argument("--conf-level", type=float, default=0.95)
    ap.add_argument("--n-bootstrap", type=int, default=300)
    ap.add_argument("--threads", type=int, default=4)
    ap.add_argument("--obs-low", type=float, default=0.0)
    ap.add_argument("--obs-high", type=float, default=500.0)
    ap.add_argument("--max-run-attempts", type=int, default=5)
    ap.add_argument(
        "--root-writer",
        choices=["auto", "uproot", "root-cli"],
        default="auto",
        help="ROOT output writer mode for generated observed data (default: auto)",
    )
    ap.add_argument("--nextstat-bin", type=Path, default=None)
    ap.add_argument(
        "--out-dir",
        type=Path,
        default=Path("bench_results/hep_dataset_bootstrap_ci_2026-02-16"),
    )
    args = ap.parse_args()

    repo_root = Path(__file__).resolve().parents[2]
    nextstat_bin = _resolve_nextstat_binary(repo_root, args.nextstat_bin)
    root_writer = _resolve_root_writer(args.root_writer)
    args.out_dir.mkdir(parents=True, exist_ok=True)

    scenarios = [
        {"name": "gauss_mu_mid", "mu_true": 125.0, "bounds": (80.0, 170.0)},
        {"name": "gauss_mu_boundary_low", "mu_true": 104.0, "bounds": (100.0, 170.0)},
    ]

    raw: dict[str, Any] = {"scenarios": {}}
    summary_scenarios: dict[str, Any] = {}

    with tempfile.TemporaryDirectory(prefix="hep_dataset_bca_") as tmp:
        tmp_dir = Path(tmp)
        macro_path: Path | None = None
        if root_writer == "root-cli":
            macro_path = tmp_dir / "write_tree_from_txt.C"
            _write_root_writer_macro(macro_path)

        for s in scenarios:
            name = str(s["name"])
            mu_true = float(s["mu_true"])
            bounds = (float(s["bounds"][0]), float(s["bounds"][1]))
            raw["scenarios"][name] = {"percentile": [], "bca": []}

            for i in range(args.runs):
                seed = args.seed0 + i
                fit_json: dict[str, Any] | None = None
                toys_json: dict[str, Any] | None = None
                fit_wall = 0.0
                toys_wall = 0.0
                obs: np.ndarray | None = None
                data_seed_used: int | None = None

                obs_root = tmp_dir / f"obs_{name}_{seed}.root"
                spec_path = tmp_dir / f"spec_{name}_{seed}.json"

                for attempt in range(args.max_run_attempts):
                    data_seed = seed + attempt * 1_000_003
                    rng = np.random.default_rng(data_seed)
                    obs_try = rng.normal(
                        loc=mu_true, scale=float(args.sigma_true), size=args.n_events
                    )
                    # Keep generated events inside observable bounds to avoid invalid fits.
                    obs_try = np.clip(
                        obs_try, float(args.obs_low) + 1e-6, float(args.obs_high) - 1e-6
                    )

                    if root_writer == "uproot":
                        _write_root_with_uproot(obs_root, obs_try)
                    else:
                        assert macro_path is not None
                        obs_txt = tmp_dir / f"obs_{name}_{seed}.txt"
                        _write_values_txt(obs_txt, obs_try)
                        _run_root_write(macro_path, obs_txt, obs_root)
                    _write_spec(
                        spec_path,
                        data_file=obs_root,
                        poi_name="gauss_mu",
                        poi_init=mu_true,
                        poi_bounds=bounds,
                        sigma=float(args.sigma_true),
                        n_events=args.n_events,
                    )

                    try:
                        fit_try, fit_wall_try = _run_json(
                            [
                                str(nextstat_bin),
                                "unbinned-fit",
                                "--config",
                                str(spec_path),
                                "--threads",
                                str(args.threads),
                            ]
                        )
                        toys_try, toys_wall_try = _run_json(
                            [
                                str(nextstat_bin),
                                "unbinned-fit-toys",
                                "--config",
                                str(spec_path),
                                "--n-toys",
                                str(args.n_bootstrap),
                                "--seed",
                                str(seed ^ 0x9E3779B97F4A7C15),
                                "--gen",
                                "mle",
                                "--threads",
                                str(args.threads),
                            ]
                        )
                    except Exception as exc:
                        if attempt + 1 == args.max_run_attempts:
                            raise RuntimeError(
                                f"run failed after {args.max_run_attempts} attempts "
                                f"(scenario={name}, seed={seed})"
                            ) from exc
                        print(
                            f"[{name}] seed={seed} retry={attempt + 1}/{args.max_run_attempts} "
                            f"(reason={exc})"
                        )
                        continue

                    fit_json = fit_try
                    toys_json = toys_try
                    fit_wall = fit_wall_try
                    toys_wall = toys_wall_try
                    obs = obs_try
                    data_seed_used = data_seed
                    break

                assert fit_json is not None
                assert toys_json is not None
                assert obs is not None
                assert data_seed_used is not None

                poi_idx = int(fit_json["poi_index"])
                obs_hat = float(fit_json["bestfit"][poi_idx])

                poi_hat = [
                    float(x)
                    for x in toys_json["results"]["poi_hat"]
                    if isinstance(x, (int, float))
                ]
                if len(poi_hat) < 2:
                    raise RuntimeError(
                        f"insufficient converged bootstrap estimates (seed={seed}, scenario={name})"
                    )

                p_lo, p_hi = _percentile_interval(poi_hat, args.conf_level)
                p_center = (p_lo + p_hi) / 2.0
                p_row = {
                    "seed": seed,
                    "data_seed": data_seed_used,
                    "mu_true": mu_true,
                    "obs_hat": obs_hat,
                    "lower": p_lo,
                    "upper": p_hi,
                    "width": p_hi - p_lo,
                    "contains_true": p_lo <= mu_true <= p_hi,
                    "center_minus_true": p_center - mu_true,
                    "obs_hat_minus_true": obs_hat - mu_true,
                    "wall_fit_s": fit_wall,
                    "wall_bootstrap_s": toys_wall,
                    "wall_total_s": fit_wall + toys_wall,
                    "effective_method": "percentile",
                    "fallback_reason": None,
                    "n_bootstrap_effective": len(poi_hat),
                }
                raw["scenarios"][name]["percentile"].append(p_row)

                a = _jackknife_acceleration_from_mean(obs)
                b_lo, b_hi, b_diag = _bca_interval(
                    bootstrap_estimates=poi_hat,
                    observed_estimate=obs_hat,
                    conf_level=args.conf_level,
                    acceleration=a,
                )
                b_center = (b_lo + b_hi) / 2.0
                b_row = {
                    "seed": seed,
                    "data_seed": data_seed_used,
                    "mu_true": mu_true,
                    "obs_hat": obs_hat,
                    "lower": b_lo,
                    "upper": b_hi,
                    "width": b_hi - b_lo,
                    "contains_true": b_lo <= mu_true <= b_hi,
                    "center_minus_true": b_center - mu_true,
                    "obs_hat_minus_true": obs_hat - mu_true,
                    "wall_fit_s": fit_wall,
                    "wall_bootstrap_s": toys_wall,
                    "wall_total_s": fit_wall + toys_wall,
                    "effective_method": b_diag["effective_method"],
                    "fallback_reason": b_diag["fallback_reason"],
                    "diagnostics": b_diag,
                    "n_bootstrap_effective": len(poi_hat),
                }
                raw["scenarios"][name]["bca"].append(b_row)

                print(
                    f"[{name}] seed={seed} obs_hat={obs_hat:.4f} "
                    f"pct_contains={p_row['contains_true']} bca_contains={b_row['contains_true']} "
                    f"bca_effective={b_row['effective_method']}"
                )

            summary_scenarios[name] = {
                "percentile": _aggregate(raw["scenarios"][name]["percentile"]),
                "bca": _aggregate(raw["scenarios"][name]["bca"]),
            }

    summary = {
        "config": {
            "runs": args.runs,
            "seed0": args.seed0,
            "n_events": args.n_events,
            "sigma_true": args.sigma_true,
            "conf_level": args.conf_level,
            "n_bootstrap": args.n_bootstrap,
            "threads": args.threads,
            "obs_low": args.obs_low,
            "obs_high": args.obs_high,
            "max_run_attempts": args.max_run_attempts,
            "root_writer": root_writer,
            "nextstat_bin": str(nextstat_bin),
        },
        "scenarios": summary_scenarios,
    }

    (args.out_dir / "raw_runs.json").write_text(json.dumps(raw, indent=2), encoding="utf-8")
    (args.out_dir / "summary.json").write_text(json.dumps(summary, indent=2), encoding="utf-8")
    _write_markdown(summary, args.out_dir / "summary.md")

    print(f"\nSaved: {args.out_dir / 'raw_runs.json'}")
    print(f"Saved: {args.out_dir / 'summary.json'}")
    print(f"Saved: {args.out_dir / 'summary.md'}")


if __name__ == "__main__":
    main()
