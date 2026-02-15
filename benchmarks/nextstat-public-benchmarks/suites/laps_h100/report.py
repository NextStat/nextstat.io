#!/usr/bin/env python3
"""LAPS H100 benchmark report generator.

Reads JSON artifacts from a results directory and generates a markdown report.

Usage:
    python report.py --results-dir /data/laps_h100_v1
"""

from __future__ import annotations

import argparse
import json
import math
import sys
from collections import defaultdict
from pathlib import Path
from typing import Any


def _fmt(v: float | None, decimals: int = 3) -> str:
    if v is None:
        return "—"
    if abs(v) >= 1e6:
        return f"{v:.1e}"
    if abs(v) >= 1000:
        return f"{v:,.0f}"
    return f"{v:.{decimals}f}"


def _fmt_speedup(v: float | None) -> str:
    if v is None:
        return "—"
    return f"{v:.2f}x"


def load_results(results_dir: Path) -> list[dict[str, Any]]:
    docs = []
    for p in sorted(results_dir.glob("*.json")):
        if p.name == "suite_index.json" or p.name.endswith("_report.json"):
            continue
        try:
            doc = json.loads(p.read_text())
            if doc.get("schema", "").startswith("nextstat.laps_h100"):
                docs.append(doc)
        except Exception:
            continue
    return docs


def group_by_prefix(docs: list[dict[str, Any]]) -> dict[str, list[dict[str, Any]]]:
    groups: dict[str, list[dict[str, Any]]] = defaultdict(list)
    for doc in docs:
        label = doc.get("label", "")
        prefix = label.split("_", 1)[0] if "_" in label else "other"
        groups[prefix].append(doc)
    return dict(groups)


def median(vals: list[float]) -> float:
    s = sorted(vals)
    n = len(s)
    if n == 0:
        return 0.0
    if n % 2 == 1:
        return s[n // 2]
    return (s[n // 2 - 1] + s[n // 2]) / 2


def aggregate_by_label(docs: list[dict[str, Any]]) -> dict[str, dict[str, Any]]:
    """Group by label (across seeds) and take median of metrics."""
    by_label: dict[str, list[dict[str, Any]]] = defaultdict(list)
    for doc in docs:
        by_label[doc["label"]].append(doc)

    agg = {}
    for label, runs in by_label.items():
        metrics_keys = ["wall_time_s", "samples_per_sec", "n_kernel_launches",
                        "min_ess_bulk", "max_r_hat", "accept_rate"]
        m: dict[str, Any] = {}
        for key in metrics_keys:
            vals = [r["metrics"][key] for r in runs if r["metrics"].get(key) is not None]
            m[key] = median(vals) if vals else None

        m["n_devices"] = runs[0]["metrics"].get("n_devices", 1)
        m["fused"] = runs[0]["metrics"].get("fused", False)
        m["n_chains"] = runs[0]["config"]["n_chains"]
        m["dim"] = runs[0]["config"]["dim"]
        m["n_seeds"] = len(runs)
        agg[label] = m
    return agg


def generate_report(docs: list[dict[str, Any]], out_path: Path) -> None:
    lines: list[str] = []
    lines.append("# LAPS H100 Benchmark Report\n")

    grouped = group_by_prefix(docs)

    # --- Weak Scaling ---
    if "weak" in grouped:
        agg = aggregate_by_label(grouped["weak"])
        lines.append("## Weak Scaling (std_normal, dim=100)\n")
        lines.append("| Config | Chains | GPUs | Wall (s) | Samples/s | ESS bulk | R-hat |")
        lines.append("|--------|--------|------|----------|-----------|----------|-------|")
        for label, m in sorted(agg.items()):
            chains = f"{m['n_chains']:,}"
            gpus = str(m["n_devices"])
            wall = _fmt(m["wall_time_s"])
            sps = _fmt(m["samples_per_sec"], 0)
            ess = _fmt(m["min_ess_bulk"], 0)
            rhat = _fmt(m["max_r_hat"], 4)
            lines.append(f"| {label} | {chains} | {gpus} | {wall} | {sps} | {ess} | {rhat} |")
        lines.append("")

    # --- Strong Scaling ---
    if "strong" in grouped:
        agg = aggregate_by_label(grouped["strong"])
        lines.append("## Strong Scaling (std_normal, dim=100, 132K chains)\n")
        lines.append("| Config | GPUs | Wall (s) | Samples/s | Speedup | R-hat |")
        lines.append("|--------|------|----------|-----------|---------|-------|")
        base_wall = None
        for label, m in sorted(agg.items()):
            wall = m["wall_time_s"]
            if base_wall is None:
                base_wall = wall
            speedup = base_wall / wall if wall and wall > 0 else None
            gpus = str(m["n_devices"])
            lines.append(f"| {label} | {gpus} | {_fmt(wall)} | {_fmt(m['samples_per_sec'], 0)} | {_fmt_speedup(speedup)} | {_fmt(m['max_r_hat'], 4)} |")
        lines.append("")

    # --- Dim Sweep ---
    if "dim" in grouped:
        agg = aggregate_by_label(grouped["dim"])
        lines.append("## Dimension Sweep (4xH100, 65K chains)\n")
        lines.append("| Dim | Wall (s) | Samples/s | ESS bulk | R-hat |")
        lines.append("|-----|----------|-----------|----------|-------|")
        for label, m in sorted(agg.items(), key=lambda x: x[1]["dim"]):
            lines.append(f"| {m['dim']} | {_fmt(m['wall_time_s'])} | {_fmt(m['samples_per_sec'], 0)} | {_fmt(m['min_ess_bulk'], 0)} | {_fmt(m['max_r_hat'], 4)} |")
        lines.append("")

    # --- Fused Ablation ---
    if "fused" in grouped:
        agg = aggregate_by_label(grouped["fused"])
        lines.append("## Fused Kernel Ablation (4xH100, 65K chains, dim=100)\n")
        lines.append("| Mode | Launches | Wall (s) | Samples/s | Speedup |")
        lines.append("|------|----------|----------|-----------|---------|")
        base_wall = None
        for label, m in sorted(agg.items()):
            wall = m["wall_time_s"]
            if base_wall is None:
                base_wall = wall
            speedup = base_wall / wall if wall and wall > 0 else None
            launches = _fmt(m["n_kernel_launches"], 0)
            lines.append(f"| {label} | {launches} | {_fmt(wall)} | {_fmt(m['samples_per_sec'], 0)} | {_fmt_speedup(speedup)} |")
        lines.append("")

    # --- Model Comparison ---
    if "model" in grouped:
        agg = aggregate_by_label(grouped["model"])
        lines.append("## Model Comparison (4xH100, 65K chains)\n")
        lines.append("| Model | Dim | Wall (s) | Samples/s | ESS bulk | R-hat | Accept |")
        lines.append("|-------|-----|----------|-----------|----------|-------|--------|")
        for label, m in sorted(agg.items()):
            accept = _fmt(m.get("accept_rate"), 3)
            lines.append(f"| {label} | {m['dim']} | {_fmt(m['wall_time_s'])} | {_fmt(m['samples_per_sec'], 0)} | {_fmt(m['min_ess_bulk'], 0)} | {_fmt(m['max_r_hat'], 4)} | {accept} |")
        lines.append("")

    report = "\n".join(lines) + "\n"
    out_path.write_text(report)
    print(f"\nReport written to: {out_path}")
    print(report)


def main() -> int:
    ap = argparse.ArgumentParser(description="LAPS H100 benchmark report generator")
    ap.add_argument("--results-dir", required=True, help="Directory with JSON artifacts")
    args = ap.parse_args()

    results_dir = Path(args.results_dir)
    docs = load_results(results_dir)
    if not docs:
        print(f"No LAPS H100 benchmark results found in {results_dir}")
        return 1

    print(f"Found {len(docs)} benchmark results")
    out_path = results_dir / "laps_h100_report.md"
    generate_report(docs, out_path)
    return 0


if __name__ == "__main__":
    sys.exit(main())
