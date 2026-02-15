#!/usr/bin/env python3
"""Monte Carlo fault-tree benchmark runner.

Single-case runner: generates a fault tree spec, runs NS + numpy baseline,
measures throughput and parity, writes a JSON artifact.

Usage:
    python run.py --case ft_bernoulli_small --workload bernoulli_fixed \
        --n-components 16 --n-scenarios 10000000 --seed 42 --out result.json
"""

from __future__ import annotations

import argparse
import hashlib
import json
import platform
import sys
import time
from pathlib import Path

import numpy as np

import nextstat

sys.path.insert(0, str(Path(__file__).resolve().parent.parent.parent))
from scripts.bench_env import collect_environment


# ---------------------------------------------------------------------------
# Spec generators
# ---------------------------------------------------------------------------

def sha256_json_obj(obj: dict) -> str:
    b = (json.dumps(obj, sort_keys=True, separators=(",", ":")) + "\n").encode("utf-8")
    return hashlib.sha256(b).hexdigest()


def make_spec_bernoulli_fixed(n_components: int, p_base: float = 0.01) -> dict:
    """Simple OR-tree with fixed Bernoulli components."""
    comps = [{"type": "bernoulli", "p": p_base * (1 + 0.5 * i / n_components)}
             for i in range(n_components)]
    nodes = [{"type": "component", "index": i} for i in range(n_components)]
    # Top = OR of all components.
    nodes.append({"type": "or", "children": list(range(n_components))})
    return {"components": comps, "nodes": nodes, "top_event": len(nodes) - 1}


def make_spec_bernoulli_uncertain(n_components: int) -> dict:
    """OR-tree with uncertain Bernoulli components (epistemic Z)."""
    comps = [{"type": "bernoulli_uncertain", "mu": -4.0 + 0.1 * i, "sigma": 0.5}
             for i in range(n_components)]
    nodes = [{"type": "component", "index": i} for i in range(n_components)]
    nodes.append({"type": "or", "children": list(range(n_components))})
    return {"components": comps, "nodes": nodes, "top_event": len(nodes) - 1}


def make_spec_weibull_mission(n_components: int) -> dict:
    """OR-tree with Weibull mission-time components."""
    comps = [{"type": "weibull_mission", "k": 1.5, "lambda": 1000.0 + 100 * i,
              "mission_time": 100.0}
             for i in range(n_components)]
    nodes = [{"type": "component", "index": i} for i in range(n_components)]
    nodes.append({"type": "or", "children": list(range(n_components))})
    return {"components": comps, "nodes": nodes, "top_event": len(nodes) - 1}


WORKLOAD_GENERATORS = {
    "bernoulli_fixed": make_spec_bernoulli_fixed,
    "bernoulli_uncertain": make_spec_bernoulli_uncertain,
    "weibull_mission": make_spec_weibull_mission,
}


# ---------------------------------------------------------------------------
# Numpy baseline
# ---------------------------------------------------------------------------


def numpy_baseline_bernoulli(spec: dict, n_scenarios: int, seed: int) -> dict:
    """Vectorized numpy baseline for Bernoulli OR-tree."""
    rng = np.random.default_rng(seed)
    n_comp = len(spec["components"])
    probs = np.array([c.get("p", 0.05) for c in spec["components"]])

    t0 = time.perf_counter()
    # Draw all at once: (n_scenarios, n_comp).
    u = rng.random((n_scenarios, n_comp))
    comp_failed = u < probs[None, :]
    # OR gate: system fails if any component fails.
    top_failed = comp_failed.any(axis=1)
    n_top = int(top_failed.sum())
    wall = time.perf_counter() - t0

    return {
        "n_top_failures": n_top,
        "p_failure": n_top / n_scenarios,
        "wall_time_s": wall,
        "scenarios_per_sec": n_scenarios / wall,
    }


def python_loop_baseline(spec: dict, n_scenarios: int, seed: int) -> dict:
    """Pure Python loop baseline (worst case, for comparison)."""
    import random
    random.seed(seed)
    n_comp = len(spec["components"])
    probs = [c.get("p", 0.05) for c in spec["components"]]

    n_top = 0
    t0 = time.perf_counter()
    for _ in range(n_scenarios):
        top_fail = False
        for j in range(n_comp):
            if random.random() < probs[j]:
                top_fail = True
                break
        if top_fail:
            n_top += 1
    wall = time.perf_counter() - t0

    return {
        "n_top_failures": n_top,
        "p_failure": n_top / n_scenarios,
        "wall_time_s": wall,
        "scenarios_per_sec": n_scenarios / wall,
    }


# ---------------------------------------------------------------------------
# Main
# ---------------------------------------------------------------------------


def main() -> int:
    ap = argparse.ArgumentParser()
    ap.add_argument("--case", required=True, help="Case ID")
    ap.add_argument("--workload", required=True, choices=list(WORKLOAD_GENERATORS))
    ap.add_argument("--n-components", type=int, default=16)
    ap.add_argument("--n-scenarios", type=int, default=10_000_000)
    ap.add_argument("--seed", type=int, default=42)
    ap.add_argument("--device", default="cpu", help="cpu or cuda")
    ap.add_argument("--chunk-size", type=int, default=0)
    ap.add_argument("--repeat", type=int, default=5, help="Timing repeats")
    ap.add_argument("--out", required=True, help="Output JSON path")
    ap.add_argument("--deterministic", action="store_true")
    ap.add_argument("--skip-baselines", action="store_true")
    args = ap.parse_args()

    spec = WORKLOAD_GENERATORS[args.workload](args.n_components)
    dataset_spec = {
        "case": str(args.case),
        "workload": str(args.workload),
        "n_components": int(args.n_components),
        "n_scenarios": int(args.n_scenarios),
        "seed": int(args.seed),
        "device": str(args.device),
        "chunk_size": int(args.chunk_size),
        "spec": spec,
    }
    dataset = {
        "id": f"generated:montecarlo_safety:{args.case}",
        "sha256": sha256_json_obj(dataset_spec),
        "spec": dataset_spec,
    }

    # --- NS runs ---
    timings = []
    last_result = None
    for i in range(args.repeat):
        r = nextstat.fault_tree_mc(
            spec,
            args.n_scenarios,
            seed=args.seed + i * (0 if args.deterministic else 1),
            device=args.device,
            chunk_size=args.chunk_size,
        )
        timings.append(r["wall_time_s"])
        last_result = r

    timings.sort()
    timing_stats = {
        "wall_time_s": {
            "min": timings[0],
            "median": timings[len(timings) // 2],
            "p95": timings[int(len(timings) * 0.95)] if len(timings) > 1 else timings[0],
        },
        "scenarios_per_sec": {
            "max": args.n_scenarios / timings[0],
            "median": args.n_scenarios / timings[len(timings) // 2],
        },
    }

    # --- Baselines ---
    baselines = {}
    if not args.skip_baselines and args.workload == "bernoulli_fixed":
        # Numpy baseline.
        np_result = numpy_baseline_bernoulli(spec, args.n_scenarios, args.seed)
        baselines["numpy_vectorized"] = np_result

        # Python loop (only for small N to not take forever).
        if args.n_scenarios <= 1_000_000:
            py_result = python_loop_baseline(spec, min(args.n_scenarios, 100_000), args.seed)
            baselines["python_loop"] = py_result

    # --- Reproducibility check ---
    if args.deterministic:
        r2 = nextstat.fault_tree_mc(
            spec, args.n_scenarios, seed=args.seed, device=args.device, chunk_size=args.chunk_size
        )
        reproducible = r2["n_top_failures"] == last_result["n_top_failures"]
    else:
        reproducible = None

    # --- Artifact ---
    artifact = {
        "schema_version": "nextstat.montecarlo_safety_benchmark_result.v1",
        "suite": "montecarlo_safety",
        "case": str(args.case),
        "deterministic": bool(args.deterministic),
        "environment": collect_environment(),
        "status": "ok",
        "meta": {
            "python": sys.version.split()[0],
            "platform": platform.platform(),
            "nextstat_version": nextstat.__version__,
        },
        "dataset": dataset,
        "config": {
            "case_id": args.case,
            "workload": args.workload,
            "n_components": args.n_components,
            "n_scenarios": args.n_scenarios,
            "seed": args.seed,
            "device": args.device,
            "chunk_size": args.chunk_size,
            "n_repeats": args.repeat,
            "deterministic": args.deterministic,
        },
        "results": {
            "p_failure": last_result["p_failure"],
            "se": last_result["se"],
            "ci_lower": last_result["ci_lower"],
            "ci_upper": last_result["ci_upper"],
            "n_top_failures": last_result["n_top_failures"],
            "component_importance": last_result["component_importance"],
        },
        "timing": timing_stats,
        "baselines": baselines,
        "reproducibility": {
            "deterministic_flag": args.deterministic,
            "bit_exact_reproduced": reproducible,
        },
    }

    with open(args.out, "w") as f:
        json.dump(artifact, f, indent=2, sort_keys=True)
        f.write("\n")

    # Print summary.
    throughput = timing_stats["scenarios_per_sec"]["median"]
    unit = "M/s" if throughput > 1e6 else "K/s"
    val = throughput / 1e6 if throughput > 1e6 else throughput / 1e3
    print(f"[{args.case}] p={last_result['p_failure']:.6f} "
          f"se={last_result['se']:.2e} "
          f"throughput={val:.1f} {unit} "
          f"wall={timing_stats['wall_time_s']['median']:.3f}s "
          f"device={args.device}")

    return 0


if __name__ == "__main__":
    raise SystemExit(main())
