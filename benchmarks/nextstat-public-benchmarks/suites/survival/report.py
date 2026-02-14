#!/usr/bin/env python3
"""Generate markdown report from survival benchmark results.

Usage:
    python report.py /tmp/survival_truth
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
    suite_path = out_dir / "survival_suite.json"

    if not suite_path.exists():
        print(f"Suite index not found: {suite_path}", file=sys.stderr)
        return 1

    suite = json.loads(suite_path.read_text())
    meta = suite.get("meta", {})
    cases = suite.get("cases", [])

    lines = [
        "# Survival Benchmark Report",
        "",
        f"**NextStat** {meta.get('nextstat_version', '?')} | "
        f"**Platform**: {meta.get('platform', '?')}",
        "",
    ]

    # --- Standard benchmark cases ---
    std_cases = [c for c in cases if not c.get("kind", "").startswith("truth_")]
    if std_cases:
        lines.append("## Standard Benchmark Cases")
        lines.append("")
        lines.append("| Case | Kind | N | Wall (s) | Parity | Status |")
        lines.append("|------|------|---|----------|--------|--------|")
        for c in std_cases:
            lines.append(
                f"| {c['case']} | {c.get('kind', '?')} | {c.get('n_subjects', '?')} | "
                f"{c.get('wall_time_median_s', 0):.4f} | {c.get('parity_status', '?')} | {c['status']} |"
            )
        lines.append("")

    # --- Truth-recovery cases ---
    truth_cases = [c for c in cases if c.get("kind", "").startswith("truth_")]
    if truth_cases:
        lines.append("## Truth-Recovery Results")
        lines.append("")
        lines.append("MC-based truth recovery: generate data from known DGP, fit NS, check bias → 0 and coverage → 95%.")
        lines.append("")

        cases_dir = out_dir / "cases"
        for c in truth_cases:
            case_path = cases_dir / f"{c['case']}.json"
            if not case_path.exists():
                continue

            data = json.loads(case_path.read_text())
            tr = data.get("truth_recovery", {})
            dgp = tr.get("dgp", {})
            n_rep = tr.get("n_replicates", 0)
            cfg = data.get("config", {})

            lines.append(f"### {c['case']}")
            lines.append("")
            lines.append(f"- **DGP**: {dgp.get('distribution', '?')}")
            lines.append(f"- **N**: {cfg.get('n', '?')}, **Replicates**: {n_rep}")
            lines.append("")

            lines.append("| Cens% | Conv% | Param | True | Mean Hat | Bias | RMSE | Coverage(95%) |")
            lines.append("|-------|-------|-------|------|----------|------|------|---------------|")

            for cr_result in tr.get("results_by_censoring", []):
                cr = cr_result["censoring_rate"]
                conv = cr_result.get("convergence_rate", 0)
                params = cr_result.get("params", [])
                for i, p in enumerate(params):
                    cens_col = f"{cr:.0%}" if i == 0 else ""
                    conv_col = f"{conv:.1%}" if i == 0 else ""
                    lines.append(
                        f"| {cens_col} | {conv_col} | {p['name']} | {p['true']:.4f} | "
                        f"{p['mean_hat']:.4f} | {p['bias']:.4f} | {p['rmse']:.4f} | "
                        f"{p['coverage_95']:.1%} |"
                    )

            lines.append("")

    # --- Summary ---
    summary = suite.get("summary", {})
    lines.append("## Summary")
    lines.append("")
    lines.append(f"- **Total cases**: {summary.get('n_cases', 0)}")
    lines.append(f"- **OK**: {summary.get('n_ok', 0)}")
    lines.append(f"- **Warn**: {summary.get('n_warn', 0)}")
    lines.append(f"- **Failed**: {summary.get('n_failed', 0)}")
    lines.append("")

    report = "\n".join(lines)
    report_path = out_dir / "report.md"
    report_path.write_text(report)
    print(report)
    print(f"\nReport written to: {report_path}")

    return 0


if __name__ == "__main__":
    raise SystemExit(main())
