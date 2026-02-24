#!/usr/bin/env python3
"""Pharma Benchmark Suite: NextStat vs open-source pharmacometric baselines.

Runs standard pharmacometric benchmark models and reports:
- Parameter estimates (theta, omega, sigma)
- Objective function value (OFV)
- Fit time (wall clock, median of 3 seeds)
- Parameter parity vs reference

Usage:
    python suite.py [--models all|warfarin|theo|pheno] [--seeds 42,123,777] [--n-repeats 3]
    python suite.py [--models all|warfarin|theo|pheno] --seed 42 [--n-repeats 3]  # legacy single-seed
    python suite.py --skip-competitors  # NS only
    python suite.py --out /tmp/pharma_bench
"""

from __future__ import annotations

import argparse
import inspect
import json
import math
import os
import sys
from pathlib import Path
from typing import Any

# Always benchmark the local extension/package from this repo.
REPO_ROOT = Path(__file__).resolve().parents[2]
# Keep benchmark R dependencies isolated/reproducible by default.
if "R_LIBS_USER" not in os.environ:
    os.environ["R_LIBS_USER"] = str(REPO_ROOT / ".r_libs")
Path(os.environ["R_LIBS_USER"]).mkdir(parents=True, exist_ok=True)

sys.path.insert(0, str(REPO_ROOT / "bindings" / "ns-py" / "python"))
sys.path.insert(0, str(REPO_ROOT))

# Environment snapshot (per benchmark-protocol.md section 6b)
sys.path.insert(0, str(REPO_ROOT / "benchmarks" / "nextstat-public-benchmarks"))
from scripts.bench_env import print_environment

from benchmarks.pharma.models import warfarin, theophylline, phenobarbital
from benchmarks.pharma import competitors, report


MODEL_REGISTRY = {
    "warfarin": warfarin,
    "theo": theophylline,
    "pheno": phenobarbital,
}


def _parse_seeds(seed_csv: str) -> list[int]:
    seeds: list[int] = []
    for token in seed_csv.split(","):
        token = token.strip()
        if not token:
            continue
        seeds.append(int(token))
    if not seeds:
        raise ValueError("seed list is empty")
    return seeds


def _median(values: list[float]) -> float | None:
    if not values:
        return None
    s = sorted(values)
    n = len(s)
    mid = n // 2
    if n % 2 == 1:
        return float(s[mid])
    return float((s[mid - 1] + s[mid]) / 2.0)


def _median_map(rows: list[dict[str, Any]]) -> dict[str, float]:
    out: dict[str, float] = {}
    keys = {k for row in rows for k in row.keys()}
    for key in sorted(keys):
        vals = [float(row[key]) for row in rows if isinstance(row.get(key), (int, float))]
        med = _median(vals)
        if med is not None:
            out[key] = med
    return out


def _theta_rse_proxy(theta_by_seed: dict[str, list[float]]) -> dict[str, float]:
    """Approximate cross-seed relative standard error: sd(theta)/|mean(theta)|."""
    out: dict[str, float] = {}
    for name, vals in theta_by_seed.items():
        if len(vals) < 2:
            continue
        mean = sum(vals) / len(vals)
        if abs(mean) <= 1e-12:
            out[name] = float("inf")
            continue
        var = sum((x - mean) ** 2 for x in vals) / (len(vals) - 1)
        sd = math.sqrt(max(var, 0.0))
        out[name] = float(sd / abs(mean))
    return out


def _aggregate_engine_runs(engine_runs: list[dict[str, Any]]) -> dict[str, Any] | None:
    if not engine_runs:
        return None

    wall_vals = [float(r["wall_s"]) for r in engine_runs if isinstance(r.get("wall_s"), (int, float))]
    ofv_vals = [float(r["ofv"]) for r in engine_runs if isinstance(r.get("ofv"), (int, float))]
    n_iter_vals = [float(r["n_iter"]) for r in engine_runs if isinstance(r.get("n_iter"), (int, float))]
    theta_rows = [r.get("theta", {}) for r in engine_runs if isinstance(r.get("theta"), dict)]
    omega_rows = [r.get("omega", {}) for r in engine_runs if isinstance(r.get("omega"), dict)]
    conv_vals = [bool(r["converged"]) for r in engine_runs if "converged" in r]

    theta_by_seed: dict[str, list[float]] = {}
    for row in theta_rows:
        for k, v in row.items():
            if isinstance(v, (int, float)):
                theta_by_seed.setdefault(k, []).append(float(v))

    out: dict[str, Any] = {
        "tool": engine_runs[0].get("tool"),
        "method": engine_runs[0].get("method"),
        "n_seeds": len(engine_runs),
        "seed_values": [r.get("seed") for r in engine_runs],
        "wall_s": _median(wall_vals),
        "wall_s_by_seed": wall_vals,
        "theta": _median_map(theta_rows),
        "theta_rse_proxy": _theta_rse_proxy(theta_by_seed),
        "omega": _median_map(omega_rows),
        "ofv": _median(ofv_vals),
        "n_iter": _median(n_iter_vals),
    }
    if conv_vals:
        out["converged_all"] = all(conv_vals)
        out["converged_rate"] = float(sum(1 for x in conv_vals if x) / len(conv_vals))
    return out


def _aggregate_model_runs(seed_runs: list[dict[str, Any]]) -> dict[str, Any]:
    first = seed_runs[0]
    out: dict[str, Any] = {
        "model": first.get("model", "unknown"),
        "description": first.get("description", ""),
        "reference": first.get("reference", {}),
        "seeds": [r.get("seed") for r in seed_runs],
        "seed_runs": seed_runs,
    }

    ns_runs = [r.get("nextstat") for r in seed_runs if isinstance(r.get("nextstat"), dict)]
    out["nextstat"] = _aggregate_engine_runs(ns_runs)

    for competitor_key in ("nlmixr2", "torsten", "saemix", "mas", "pharmpy"):
        comp_runs = [r.get(competitor_key) for r in seed_runs if isinstance(r.get(competitor_key), dict)]
        out[competitor_key] = _aggregate_engine_runs(comp_runs)

    return out


def _pre_run_check() -> dict:
    """Pre-run checklist: verify NS Python API imports and pharma functions exist.

    Per benchmark-protocol.md section 3.
    """
    print("\n--- Pre-run checklist ---")

    # 1. Verify NS importable
    try:
        import nextstat
        print(f"  [ok] nextstat {nextstat.__version__} imported")
    except ImportError as e:
        print(f"  [FAIL] nextstat import failed: {e}")
        sys.exit(1)

    module_path = Path(getattr(nextstat, "__file__", "")).resolve()
    expected_root = (REPO_ROOT / "bindings" / "ns-py" / "python").resolve()
    if expected_root not in module_path.parents and module_path != expected_root:
        print(f"  [FAIL] nextstat module is not local repo build: {module_path}")
        print(f"        expected under: {expected_root}")
        sys.exit(1)
    print(f"  [ok] nextstat module path: {module_path}")

    # 2. Verify nlme_foce exists
    if nextstat.nlme_foce is None:
        print("  [FAIL] nextstat.nlme_foce is None (not built)")
        sys.exit(1)
    print("  [ok] nextstat.nlme_foce available")
    foce_sig = inspect.signature(nextstat.nlme_foce)
    if "doses" not in foce_sig.parameters:
        print(f"  [FAIL] nlme_foce API mismatch: expected `doses` kwarg, got {foce_sig}")
        sys.exit(1)
    print("  [ok] nlme_foce uses doses= API")

    # 3. Verify nlme_saem exists
    if nextstat.nlme_saem is None:
        print("  [warn] nextstat.nlme_saem is None (not built)")
    else:
        print("  [ok] nextstat.nlme_saem available")
        saem_sig = inspect.signature(nextstat.nlme_saem)
        if "doses" not in saem_sig.parameters:
            print(f"  [FAIL] nlme_saem API mismatch: expected `doses` kwarg, got {saem_sig}")
            sys.exit(1)
        if "model" not in saem_sig.parameters:
            print(f"  [FAIL] nlme_saem API mismatch: missing `model` kwarg, got {saem_sig}")
            sys.exit(1)
        print("  [ok] nlme_saem uses model= + doses= API")

    # 4. Check competitors
    print(f"  [ok] R_LIBS_USER={os.environ.get('R_LIBS_USER')}")
    avail = competitors.list_available()
    for name, version in avail.items():
        if version is not None:
            print(f"  [ok] {name} {version} available")
        else:
            print(f"  [skip] {name} not installed")

    print("--- Pre-run checklist complete ---\n")
    return {
        "nextstat_version": str(getattr(nextstat, "__version__", "unknown")),
        "nextstat_module": str(module_path),
        "nlme_foce_signature": str(foce_sig),
        "nlme_saem_signature": str(inspect.signature(nextstat.nlme_saem))
        if nextstat.nlme_saem is not None else None,
        "competitors_available": avail,
    }


def main():
    ap = argparse.ArgumentParser(
        description="Pharma Benchmark Suite: NextStat vs pharmacometric competitors"
    )
    ap.add_argument(
        "--models", default="all",
        help="Comma-separated model names or 'all'. Options: warfarin, theo, pheno"
    )
    ap.add_argument("--seed", type=int, default=None,
                     help="Legacy single seed override (disables --seeds sweep)")
    ap.add_argument("--seeds", default="42,123,777",
                     help="Comma-separated seed list for regulatory-grade aggregation")
    ap.add_argument("--n-repeats", type=int, default=3,
                     help="Number of timing repeats (median taken)")
    ap.add_argument("--skip-competitors", action="store_true",
                     help="Skip competitor benchmarks (NS only)")
    ap.add_argument("--out", default="/tmp/pharma_bench",
                     help="Output directory for JSON artifacts")
    args = ap.parse_args()

    # Parse model list
    if args.models == "all":
        model_names = list(MODEL_REGISTRY.keys())
    else:
        model_names = [m.strip() for m in args.models.split(",")]
        for m in model_names:
            if m not in MODEL_REGISTRY:
                print(f"Unknown model: {m}. Available: {list(MODEL_REGISTRY.keys())}")
                sys.exit(1)

    if args.seed is not None:
        seeds = [int(args.seed)]
    else:
        try:
            seeds = _parse_seeds(args.seeds)
        except Exception as e:
            print(f"Invalid --seeds value '{args.seeds}': {e}")
            sys.exit(1)

    out_dir = Path(args.out)
    out_dir.mkdir(parents=True, exist_ok=True)

    # Collect and print environment (mandatory per protocol section 6b)
    env = print_environment()

    # Pre-run checklist
    harness_check = _pre_run_check()

    # Run benchmarks
    print(f"\n{'='*70}")
    print(f"  PHARMA BENCHMARK SUITE")
    print(f"  Models: {', '.join(model_names)}")
    print(f"  Seeds: {seeds}  |  Repeats: {args.n_repeats}")
    print(f"{'='*70}")

    all_results = []

    for model_name in model_names:
        run_fn = MODEL_REGISTRY[model_name]
        print(f"\n{'='*70}")
        print(f"  Model: {model_name}")
        print(f"{'='*70}")

        seed_runs: list[dict[str, Any]] = []
        for seed in seeds:
            print(f"  -> seed={seed}")
            try:
                result = run_fn(
                    seed=seed,
                    n_repeats=args.n_repeats,
                    skip_competitors=args.skip_competitors,
                )
                result["seed"] = seed
                seed_runs.append(result)
            except Exception as e:
                print(f"  FAILED seed={seed}: {e}")
                import traceback
                traceback.print_exc()
                seed_runs.append({
                    "model": model_name,
                    "seed": seed,
                    "error": str(e),
                })

        all_results.append(_aggregate_model_runs(seed_runs))

    # Print summary table (per benchmark protocol)
    report.print_summary_table(all_results)

    # Print detailed parity report
    report.print_detailed_report(all_results)

    # Save JSON artifact with environment snapshot (mandatory per protocol section 6b)
    artifact = {
        "environment": env,
        "config": {
            "models": model_names,
            "seed": args.seed,
            "seeds": seeds,
            "n_seeds": len(seeds),
            "n_repeats": args.n_repeats,
            "skip_competitors": args.skip_competitors,
        },
        "harness_check": harness_check,
        "results": all_results,
    }

    # Clean non-serializable values
    def _clean(obj):
        if isinstance(obj, dict):
            return {k: _clean(v) for k, v in obj.items()}
        if isinstance(obj, list):
            return [_clean(v) for v in obj]
        if isinstance(obj, float):
            if not __import__("math").isfinite(obj):
                return str(obj)
        return obj

    artifact = _clean(artifact)

    out_path = out_dir / "pharma_bench.json"
    with open(out_path, "w") as f:
        json.dump(artifact, f, indent=2, default=str)
    print(f"\nJSON artifact: {out_path}")


if __name__ == "__main__":
    main()
