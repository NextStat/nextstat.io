#!/usr/bin/env python3
"""LAPS H100 benchmark runner — single case execution.

Runs a single LAPS GPU benchmark case and writes a JSON artifact.

Usage:
    python run.py --label dim100_4xH100 --model std_normal --dim 100 \
        --n-chains 65536 --devices 0,1,2,3 --seed 42 --out results/

    python run.py --label fused_1000 --model std_normal --dim 100 \
        --n-chains 65536 --fused 1000 --seed 42 --out results/
"""

from __future__ import annotations

import argparse
import json
import math
import platform
import sys
import time
from pathlib import Path
from typing import Any

sys.path.insert(0, str(Path(__file__).resolve().parent.parent.parent))
from scripts.bench_env import collect_environment, print_environment


SCHEMA = "nextstat.laps_h100_benchmark_result.v1"


def _safe_float(x: Any) -> float | None:
    try:
        v = float(x)
    except Exception:
        return None
    if not math.isfinite(v):
        return None
    return v


def _write_json(path: Path, doc: dict[str, Any]) -> None:
    path.write_text(json.dumps(doc, indent=2, sort_keys=True) + "\n")


def _diag_summary(diag: dict[str, Any]) -> dict[str, float | None]:
    r_hat = diag.get("r_hat") if isinstance(diag.get("r_hat"), dict) else {}
    ess_bulk = diag.get("ess_bulk") if isinstance(diag.get("ess_bulk"), dict) else {}
    ess_tail = diag.get("ess_tail") if isinstance(diag.get("ess_tail"), dict) else {}

    rhat_vals = [v for v in (_safe_float(x) for x in r_hat.values()) if v is not None]
    ess_bulk_vals = [v for v in (_safe_float(x) for x in ess_bulk.values()) if v is not None]
    ess_tail_vals = [v for v in (_safe_float(x) for x in ess_tail.values()) if v is not None]

    return {
        "max_r_hat": max(rhat_vals) if rhat_vals else None,
        "min_ess_bulk": min(ess_bulk_vals) if ess_bulk_vals else None,
        "min_ess_tail": min(ess_tail_vals) if ess_tail_vals else None,
    }


def run_laps(
    *,
    model: str,
    dim: int,
    n_chains: int,
    n_warmup: int,
    n_samples: int,
    seed: int,
    devices: list[int] | None,
    target_accept: float,
    sync_interval: int,
    welford_chains: int,
    batch_size: int,
    fused: int,
) -> dict[str, Any]:
    import nextstat

    model_data = {"dim": dim} if model in ("std_normal", "neal_funnel") else None
    if model == "eight_schools":
        model_data = {
            "y": [28.0, 8.0, -3.0, 7.0, -1.0, 1.0, 18.0, 12.0],
            "sigma": [15.0, 10.0, 16.0, 11.0, 9.0, 11.0, 10.0, 18.0],
        }

    t0 = time.perf_counter()
    result = nextstat.sample_laps(
        model,
        model_data=model_data,
        n_chains=n_chains,
        n_warmup=n_warmup,
        n_samples=n_samples,
        seed=seed,
        target_accept=target_accept,
        device_ids=devices,
        sync_interval=sync_interval,
        welford_chains=welford_chains,
        batch_size=batch_size,
        fused_transitions=fused,
    )
    wall_s = time.perf_counter() - t0

    diag = result.get("diagnostics", {})
    ds = _diag_summary(diag)

    phase_times = {
        "init": _safe_float(result.get("phase_times", [0, 0, 0])[0]) if "phase_times" in result else None,
        "warmup": _safe_float(result.get("phase_times", [0, 0, 0])[1]) if "phase_times" in result else None,
        "sampling": _safe_float(result.get("phase_times", [0, 0, 0])[2]) if "phase_times" in result else None,
    }

    # Acceptance rate from sample_stats
    sample_stats = result.get("sample_stats", {})
    accept_probs = sample_stats.get("accept_prob", [])
    if isinstance(accept_probs, dict):
        all_probs = []
        for chain_probs in accept_probs.values():
            if isinstance(chain_probs, list):
                all_probs.extend(chain_probs)
        accept_rate = sum(all_probs) / len(all_probs) if all_probs else None
    else:
        accept_rate = None

    n_kernel_launches = result.get("n_kernel_launches", 0)
    n_devices = result.get("n_devices", 1)
    sampling_time = phase_times.get("sampling") or wall_s
    samples_per_sec = (n_chains * n_samples / sampling_time) if sampling_time > 0 else 0

    return {
        "wall_time_s": wall_s,
        "phase_times": phase_times,
        "n_kernel_launches": n_kernel_launches,
        "samples_per_sec": samples_per_sec,
        "min_ess_bulk": ds["min_ess_bulk"],
        "max_r_hat": ds["max_r_hat"],
        "min_ess_tail": ds["min_ess_tail"],
        "accept_rate": accept_rate,
        "n_devices": n_devices,
        "fused": fused > 0,
    }


def main() -> int:
    ap = argparse.ArgumentParser(description="LAPS H100 single-case benchmark runner")
    ap.add_argument("--label", required=True, help="Case label (e.g. dim100_4xH100)")
    ap.add_argument("--model", default="std_normal",
                    choices=["std_normal", "eight_schools", "neal_funnel", "glm_logistic"])
    ap.add_argument("--dim", type=int, default=100)
    ap.add_argument("--n-chains", type=int, default=65536)
    ap.add_argument("--n-warmup", type=int, default=500)
    ap.add_argument("--n-samples", type=int, default=2000)
    ap.add_argument("--seed", type=int, default=42)
    ap.add_argument("--devices", default=None,
                    help="Comma-separated device IDs (default: auto-detect)")
    ap.add_argument("--target-accept", type=float, default=0.9)
    ap.add_argument("--sync-interval", type=int, default=100)
    ap.add_argument("--welford-chains", type=int, default=256)
    ap.add_argument("--batch-size", type=int, default=1000)
    ap.add_argument("--fused", type=int, default=0,
                    help="Fused transitions per kernel launch (0 = batched)")
    ap.add_argument("--out", required=True, help="Output directory")
    args = ap.parse_args()

    devices = [int(d) for d in args.devices.split(",")] if args.devices else None

    print(f"[LAPS H100] {args.label}: model={args.model}, dim={args.dim}, "
          f"chains={args.n_chains}, devices={devices or 'auto'}, fused={args.fused}")

    out_dir = Path(args.out)
    out_dir.mkdir(parents=True, exist_ok=True)

    metrics = run_laps(
        model=args.model,
        dim=args.dim,
        n_chains=args.n_chains,
        n_warmup=args.n_warmup,
        n_samples=args.n_samples,
        seed=args.seed,
        devices=devices,
        target_accept=args.target_accept,
        sync_interval=args.sync_interval,
        welford_chains=args.welford_chains,
        batch_size=args.batch_size,
        fused=args.fused,
    )

    env = print_environment()

    doc = {
        "schema": SCHEMA,
        "label": args.label,
        "config": {
            "model": args.model,
            "dim": args.dim,
            "n_chains": args.n_chains,
            "n_warmup": args.n_warmup,
            "n_samples": args.n_samples,
            "seed": args.seed,
            "devices": devices,
            "target_accept": args.target_accept,
            "sync_interval": args.sync_interval,
            "welford_chains": args.welford_chains,
            "batch_size": args.batch_size,
            "fused": args.fused,
        },
        "metrics": metrics,
        "environment": env,
    }

    fname = f"{args.label}_seed{args.seed}.json"
    _write_json(out_dir / fname, doc)
    print(f"  -> {out_dir / fname}")
    print(f"  wall={metrics['wall_time_s']:.3f}s, "
          f"samples/s={metrics['samples_per_sec']:.0f}, "
          f"launches={metrics['n_kernel_launches']}, "
          f"ESS_bulk={metrics['min_ess_bulk']}, "
          f"R-hat={metrics['max_r_hat']}")
    return 0


if __name__ == "__main__":
    sys.exit(main())
