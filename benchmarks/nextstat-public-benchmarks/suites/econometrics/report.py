#!/usr/bin/env python3
"""Small report generator for the econometrics suite results.

This is human-facing only; machine-readable artifacts are the JSON results.
"""

from __future__ import annotations

import argparse
import json
from pathlib import Path


def load_json(path: Path) -> dict:
    return json.loads(path.read_text())


def main() -> int:
    ap = argparse.ArgumentParser()
    ap.add_argument("--suite", required=True, help="Path to econometrics_suite.json")
    ap.add_argument("--out", required=True, help="Output markdown path.")
    args = ap.parse_args()

    suite_path = Path(args.suite).resolve()
    out_path = Path(args.out).resolve()
    out_path.parent.mkdir(parents=True, exist_ok=True)

    obj = load_json(suite_path)
    cases = obj.get("cases", []) or []
    lines: list[str] = []
    lines.append("# Econometrics Suite (Snapshot Snippet)\n")
    lines.append(f"- suite: `{obj.get('suite','')}`")
    meta = obj.get("meta", {}) or {}
    lines.append(f"- nextstat: `{meta.get('nextstat_version','')}`")
    lines.append(f"- python: `{meta.get('python','')}`")
    lines.append(f"- platform: `{meta.get('platform','')}`\n")
    lines.append("## Cases\n")
    for c in cases:
        case = str(c.get("case", ""))
        kind = str(c.get("kind", ""))
        status = str(c.get("status", ""))
        parity = str(c.get("parity_status", ""))
        n_obs = int(c.get("n_obs", 0) or 0)
        med = float(c.get("wall_time_median_s", 0.0) or 0.0)

        base_name = None
        base_med = None
        speedup = None
        try:
            rel_path = c.get("path")
            if isinstance(rel_path, str) and rel_path:
                case_obj = load_json(suite_path.parent / rel_path)
                tb = case_obj.get("timing_baseline", {}) or {}
                if isinstance(tb, dict):
                    base_name = tb.get("name")
                    base_med = (tb.get("wall_time_s", {}) or {}).get("median")
                    try:
                        base_med = float(base_med) if base_med is not None else None
                    except Exception:
                        base_med = None
        except Exception:
            pass

        if base_med is not None and med > 0:
            speedup = base_med / med

        base_str = (
            f", baseline={base_name} {base_med:.6f}s, speedup={speedup:.2f}x"
            if (base_name is not None and base_med is not None and speedup is not None)
            else ""
        )
        lines.append(
            f"- `{case}` ({kind}): status={status}, parity={parity}, n_obs={n_obs}, ns_median={med:.6f}s{base_str}"
        )

    out_path.write_text("\n".join(lines) + "\n")
    return 0


if __name__ == "__main__":
    raise SystemExit(main())
