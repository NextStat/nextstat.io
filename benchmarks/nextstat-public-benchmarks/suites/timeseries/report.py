#!/usr/bin/env python3
"""Small report generator for the timeseries suite results (human-facing)."""

from __future__ import annotations

import argparse
import json
from pathlib import Path


def _load_json(path: Path) -> dict:
    return json.loads(path.read_text())


def main() -> int:
    ap = argparse.ArgumentParser()
    ap.add_argument("--suite", required=True, help="Path to timeseries_suite.json")
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
    lines.append("# Timeseries Suite (Snapshot)\n")
    lines.append(f"- suite: `{obj.get('suite','')}`")
    lines.append(f"- nextstat: `{meta.get('nextstat_version','')}`")
    lines.append(f"- python: `{meta.get('python','')}`")
    lines.append(f"- platform: `{meta.get('platform','')}`\n")

    lines.append("## Summary\n")
    lines.append(f"- Total cases: {summary.get('n_cases', 0)}")
    lines.append(f"- OK: {summary.get('n_ok', 0)}")
    lines.append(f"- Warnings: {summary.get('n_warn', 0)}")
    lines.append(f"- Failed: {summary.get('n_failed', 0)}")
    lines.append(f"- Worst case: `{summary.get('worst_case', '')}`\n")

    lines.append("## Cases\n")
    lines.append("| Case | Kind | N | Status | Parity | Median (s) |")
    lines.append("|---|---|---:|---|---|---:|")
    for c in cases:
        lines.append(
            f"| `{c.get('case','')}` | {c.get('kind','')} | {int(c.get('n', 0) or 0)} | "
            f"{c.get('status','')} | {c.get('parity_status','')} | {float(c.get('wall_time_median_s', 0.0) or 0.0):.6f} |"
        )

    out_path.write_text("\n".join(lines) + "\n")
    return 0


if __name__ == "__main__":
    raise SystemExit(main())

