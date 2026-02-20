#!/usr/bin/env python3
"""Pharma Benchmark Suite: NextStat vs NONMEM/nlmixr2/Pumas.

Runs standard pharmacometric benchmark models and reports:
- Parameter estimates (theta, omega, sigma)
- Objective function value (OFV)
- Fit time (wall clock, median of 3 seeds)
- Parameter parity vs reference

Usage:
    python suite.py [--models all|warfarin|theo|pheno] [--seed 42] [--n-repeats 3]
    python suite.py --skip-competitors  # NS only
    python suite.py --out /tmp/pharma_bench
"""

from __future__ import annotations

import argparse
import json
import sys
import time
from pathlib import Path

# Always benchmark the local extension/package from this repo.
REPO_ROOT = Path(__file__).resolve().parents[2]
PHARMA_DIR = Path(__file__).resolve().parent
sys.path.insert(0, str(REPO_ROOT / "bindings" / "ns-py" / "python"))
sys.path.insert(0, str(REPO_ROOT))

# Environment snapshot (per benchmark-protocol.md section 6b)
sys.path.insert(0, str(REPO_ROOT / "benchmarks" / "nextstat-public-benchmarks"))
from scripts.bench_env import collect_environment, print_environment

from benchmarks.pharma.models import warfarin, theophylline, phenobarbital
from benchmarks.pharma import competitors, report


MODEL_REGISTRY = {
    "warfarin": warfarin,
    "theo": theophylline,
    "pheno": phenobarbital,
}


def _pre_run_check():
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

    # 2. Verify nlme_foce exists
    if nextstat.nlme_foce is None:
        print("  [FAIL] nextstat.nlme_foce is None (not built)")
        sys.exit(1)
    print("  [ok] nextstat.nlme_foce available")

    # 3. Verify nlme_saem exists
    if nextstat.nlme_saem is None:
        print("  [warn] nextstat.nlme_saem is None (not built)")
    else:
        print("  [ok] nextstat.nlme_saem available")

    # 4. Check competitors
    avail = competitors.list_available()
    for name, version in avail.items():
        if version is not None:
            print(f"  [ok] {name} {version} available")
        else:
            print(f"  [skip] {name} not installed")

    print("--- Pre-run checklist complete ---\n")


def main():
    ap = argparse.ArgumentParser(
        description="Pharma Benchmark Suite: NextStat vs pharmacometric competitors"
    )
    ap.add_argument(
        "--models", default="all",
        help="Comma-separated model names or 'all'. Options: warfarin, theo, pheno"
    )
    ap.add_argument("--seed", type=int, default=42)
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

    out_dir = Path(args.out)
    out_dir.mkdir(parents=True, exist_ok=True)

    # Collect and print environment (mandatory per protocol section 6b)
    env = print_environment()

    # Pre-run checklist
    _pre_run_check()

    # Run benchmarks
    print(f"\n{'='*70}")
    print(f"  PHARMA BENCHMARK SUITE")
    print(f"  Models: {', '.join(model_names)}")
    print(f"  Seed: {args.seed}  |  Repeats: {args.n_repeats}")
    print(f"{'='*70}")

    all_results = []

    for model_name in model_names:
        run_fn = MODEL_REGISTRY[model_name]
        print(f"\n{'='*70}")
        print(f"  Model: {model_name}")
        print(f"{'='*70}")

        try:
            result = run_fn(seed=args.seed, n_repeats=args.n_repeats)
            all_results.append(result)
        except Exception as e:
            print(f"  FAILED: {e}")
            import traceback
            traceback.print_exc()
            all_results.append({
                "model": model_name,
                "error": str(e),
            })

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
            "n_repeats": args.n_repeats,
            "skip_competitors": args.skip_competitors,
        },
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
