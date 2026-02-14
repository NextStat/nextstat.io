#!/usr/bin/env python3
"""Report generator for the GLM suite results.

Human-facing only; machine-readable artifacts are the JSON results.
"""

from __future__ import annotations

import argparse
import json
from pathlib import Path


def load_json(path: Path) -> dict:
    return json.loads(path.read_text())


def main() -> int:
    ap = argparse.ArgumentParser()
    ap.add_argument("--suite", required=True, help="Path to glm_suite.json")
    ap.add_argument("--out", required=True, help="Output markdown path")
    ap.add_argument("--detail", action="store_true", help="Include per-competitor parity details")
    args = ap.parse_args()

    suite_path = Path(args.suite).resolve()
    out_path = Path(args.out).resolve()
    out_path.parent.mkdir(parents=True, exist_ok=True)

    obj = load_json(suite_path)
    cases = obj.get("cases", []) or []
    lines: list[str] = []
    lines.append("# GLM Suite (Snapshot)\n")
    meta = obj.get("meta", {}) or {}
    lines.append(f"- suite: `{obj.get('suite', '')}`")
    lines.append(f"- nextstat: `{meta.get('nextstat_version', '')}`")
    lines.append(f"- python: `{meta.get('python', '')}`")
    lines.append(f"- platform: `{meta.get('platform', '')}`\n")

    summary = obj.get("summary", {}) or {}
    lines.append("## Summary\n")
    lines.append(f"- Total cases: {summary.get('n_cases', 0)}")
    lines.append(f"- OK: {summary.get('n_ok', 0)}")
    lines.append(f"- Warnings: {summary.get('n_warn', 0)}")
    lines.append(f"- Failed: {summary.get('n_failed', 0)}")
    lines.append(f"- Slowest case: `{summary.get('worst_case', 'none')}`\n")

    lines.append("## Cases\n")
    lines.append("| Case | Family | N | Status | Parity | Converged | Median (s) |")
    lines.append("|------|--------|---|--------|--------|-----------|------------|")
    for c in cases:
        case = str(c.get("case", ""))
        family = str(c.get("family", ""))
        n = int(c.get("n", 0) or 0)
        status = str(c.get("status", ""))
        parity = str(c.get("parity_status", ""))
        converged = "yes" if c.get("converged", False) else "no"
        med = float(c.get("wall_time_median_s", 0.0) or 0.0)
        lines.append(f"| `{case}` | {family} | {n:,} | {status} | {parity} | {converged} | {med:.6f} |")

    if args.detail:
        lines.append("\n## Parity Details\n")
        suite_dir = suite_path.parent
        for c in cases:
            case = str(c.get("case", ""))
            case_path = suite_dir / str(c.get("path", ""))
            if not case_path.exists():
                continue
            case_obj = load_json(case_path)
            parity_obj = case_obj.get("parity", {})
            if not isinstance(parity_obj, dict) or not parity_obj:
                continue
            lines.append(f"### `{case}`\n")
            for comp, pdata in sorted(parity_obj.items()):
                if not isinstance(pdata, dict):
                    continue
                ref = pdata.get("reference", {})
                ref_name = str(ref.get("name", comp))
                ref_ver = str(ref.get("version", ""))
                pstatus = str(pdata.get("status", ""))
                metrics = pdata.get("metrics", {})
                lines.append(f"- **{ref_name}** ({ref_ver}): {pstatus}")
                if isinstance(metrics, dict):
                    for mk, mv in sorted(metrics.items()):
                        if mv is not None:
                            lines.append(f"  - {mk}: {mv:.2e}" if isinstance(mv, float) else f"  - {mk}: {mv}")
                error = pdata.get("error")
                if error:
                    lines.append(f"  - error: {error}")

    out_path.write_text("\n".join(lines) + "\n")
    print(f"Report written to {out_path}")
    return 0


if __name__ == "__main__":
    raise SystemExit(main())
