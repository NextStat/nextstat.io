#!/usr/bin/env python3
"""Small report generator for the insurance suite results (human-facing)."""

from __future__ import annotations

import argparse
import json
from pathlib import Path


def _load_json(path: Path) -> dict:
    return json.loads(path.read_text())


def _fmt_opt(x) -> str:
    if x is None:
        return "—"
    try:
        return f"{float(x):.6g}"
    except Exception:
        return "—"


def main() -> int:
    ap = argparse.ArgumentParser()
    ap.add_argument("--suite", required=True, help="Path to insurance_suite.json")
    ap.add_argument("--out", required=True, help="Output markdown path.")
    args = ap.parse_args()

    suite_path = Path(args.suite).resolve()
    out_path = Path(args.out).resolve()
    out_path.parent.mkdir(parents=True, exist_ok=True)

    obj = _load_json(suite_path)
    meta = obj.get("meta", {}) or {}
    cases = obj.get("cases", []) or []
    summary = obj.get("summary", {}) or {}

    lines: list[str] = []
    lines.append("# Insurance Suite (Snapshot)\n")
    lines.append(f"- suite: `{obj.get('suite','')}`")
    lines.append(f"- python: `{meta.get('python','')}`")
    lines.append(f"- platform: `{meta.get('platform','')}`\n")

    lines.append("## Summary\n")
    lines.append(f"- Total cases: {summary.get('n_cases', 0)}")
    lines.append(f"- OK: {summary.get('n_ok', 0)}")
    lines.append(f"- Warnings: {summary.get('n_warn', 0)}")
    lines.append(f"- Failed: {summary.get('n_failed', 0)}")
    lines.append(f"- Worst case: `{summary.get('worst_case', '')}`\n")

    lines.append("## Cases\n")
    lines.append("| Case | Kind | Triangle | Status | Parity | Median (s) | speedup | max rel diff ultimates | rel diff IBNR |")
    lines.append("|---|---|---:|---|---|---:|---:|---:|---:|")
    for c in cases:
        lines.append(
            f"| `{c.get('case','')}` | {c.get('kind','')} | {int(c.get('triangle_size', 0) or 0)} | {c.get('status','')} | "
            f"{c.get('parity_status','')} | {float(c.get('wall_time_median_s_nextstat', 0.0) or 0.0):.6f} | "
            f"{_fmt_opt(c.get('speedup_vs_chainladder'))} | {_fmt_opt(c.get('ultimates_max_rel_diff'))} | {_fmt_opt(c.get('total_ibnr_rel_diff'))} |"
        )

    out_path.write_text("\n".join(lines) + "\n")
    return 0


if __name__ == "__main__":
    raise SystemExit(main())

