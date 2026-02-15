#!/usr/bin/env python3
"""Generate markdown report from MC safety benchmark results.

Usage:
    python report.py /tmp/mc_safety
"""

from __future__ import annotations

import json
import sys
from pathlib import Path


def main() -> int:
    if len(sys.argv) < 2:
        print("Usage: python report.py <out_dir>", file=sys.stderr)
        return 1

    out_dir = Path(sys.argv[1])
    suite_path = out_dir / "montecarlo_safety_suite.json"

    if not suite_path.exists():
        print(f"Suite index not found: {suite_path}", file=sys.stderr)
        return 1

    suite = json.loads(suite_path.read_text())
    meta = suite.get("meta", {})

    lines = [
        "# Monte Carlo Safety Benchmark Report",
        "",
        f"**NextStat** {meta.get('nextstat_version', '?')} | "
        f"**Platform**: {meta.get('platform', '?')} | "
        f"**Device**: {suite.get('device', 'cpu')}",
        "",
        "## Throughput Summary",
        "",
        "| Case | Workload | Components | Scenarios | Throughput (M/s) | Status |",
        "|------|----------|-----------|-----------|-----------------|--------|",
    ]

    for c in suite.get("cases", []):
        throughput = c.get("throughput_median", 0)
        throughput_m = throughput / 1e6 if throughput else 0
        lines.append(
            f"| {c['case']} | {c['workload']} | {c['n_components']} | "
            f"{c['n_scenarios']:,} | {throughput_m:.1f} | {c['status']} |"
        )

    lines.append("")

    # Detailed per-case results.
    lines.append("## Detailed Results")
    lines.append("")

    for c in suite.get("cases", []):
        rel = c.get("path")
        if isinstance(rel, str) and rel.strip():
            case_path = out_dir / rel
        else:
            # Back-compat for older artifacts.
            case_path = (out_dir / "cases" / f"{c['case']}.json")
        if not case_path.exists():
            continue

        data = json.loads(case_path.read_text())
        results = data.get("results", {})
        timing = data.get("timing", {})
        baselines = data.get("baselines", {})
        repro = data.get("reproducibility", {})

        lines.append(f"### {c['case']}")
        lines.append("")
        lines.append(f"- **P(failure)**: {results.get('p_failure', 0):.6f} "
                     f"(SE: {results.get('se', 0):.2e})")
        lines.append(f"- **95% CI**: [{results.get('ci_lower', 0):.6f}, "
                     f"{results.get('ci_upper', 0):.6f}]")

        wall = timing.get("wall_time_s", {})
        thru = timing.get("scenarios_per_sec", {})
        lines.append(f"- **Wall time**: {wall.get('median', 0):.3f}s "
                     f"(min: {wall.get('min', 0):.3f}s)")
        lines.append(f"- **Throughput**: {thru.get('median', 0)/1e6:.1f} M/s "
                     f"(peak: {thru.get('max', 0)/1e6:.1f} M/s)")

        if repro.get("bit_exact_reproduced") is not None:
            lines.append(f"- **Reproducible**: {repro['bit_exact_reproduced']}")

        # Baselines comparison.
        if baselines:
            lines.append("")
            lines.append("| Baseline | Throughput (M/s) | Speedup |")
            lines.append("|----------|-----------------|---------|")
            ns_thru = thru.get("median", 1)
            for name, bl in baselines.items():
                bl_thru = bl.get("scenarios_per_sec", 1)
                speedup = ns_thru / bl_thru if bl_thru > 0 else float("inf")
                lines.append(f"| {name} | {bl_thru/1e6:.1f} | {speedup:.1f}x |")

        lines.append("")

    # GPU vs CPU comparison: if multiple suite files exist, merge them.
    gpu_suite_paths = sorted(out_dir.glob("montecarlo_safety_suite_*.json"))
    if gpu_suite_paths:
        lines.append("## GPU vs CPU Comparison")
        lines.append("")
        lines.append("| Workload | Backend | Components | Scenarios | Throughput (M/s) | Speedup vs CPU |")
        lines.append("|----------|---------|-----------|-----------|-----------------|----------------|")

        # Build CPU throughput lookup.
        cpu_throughput = {}
        for c in suite.get("cases", []):
            cpu_throughput[c["case"]] = c.get("throughput_median", 0)

        for gpath in gpu_suite_paths:
            gsuite = json.loads(gpath.read_text())
            gdev = gsuite.get("device", "gpu")
            for gc in gsuite.get("cases", []):
                g_thru = gc.get("throughput_median", 0)
                g_thru_m = g_thru / 1e6 if g_thru else 0
                cpu_thru = cpu_throughput.get(gc["case"], 0)
                speedup = g_thru / cpu_thru if cpu_thru > 0 else 0
                lines.append(
                    f"| {gc['workload']} | {gdev} | {gc['n_components']} | "
                    f"{gc['n_scenarios']:,} | {g_thru_m:.1f} | {speedup:.1f}x |"
                )

        lines.append("")

    report = "\n".join(lines)
    report_path = out_dir / "report.md"
    report_path.write_text(report)
    print(report)
    print(f"\nReport written to: {report_path}")

    return 0


if __name__ == "__main__":
    raise SystemExit(main())
