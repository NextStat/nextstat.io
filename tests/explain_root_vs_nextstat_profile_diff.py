#!/usr/bin/env python3
"""Explain ROOT vs NextStat profile-scan differences from saved artifacts.

This script is meant to be run *after* `tests/validate_root_profile_scan.py` has
produced a `run_<timestamp>/` directory containing:
  - summary.json
  - root_profile_scan.json
  - nextstat_profile_scan.json

It does not require ROOT, only Python.

Run:
  ./.venv/bin/python tests/explain_root_vs_nextstat_profile_diff.py --run-dir tmp/root_parity/run_...
"""

from __future__ import annotations

import argparse
import json
import math
from pathlib import Path
from typing import Any, Dict, List, Tuple


def _load_json(path: Path) -> Dict[str, Any]:
    return json.loads(path.read_text())


def _topk(items: List[Tuple[float, float]], k: int) -> List[Tuple[float, float]]:
    return sorted(items, key=lambda t: abs(t[1]), reverse=True)[:k]


def main() -> int:
    ap = argparse.ArgumentParser()
    src = ap.add_mutually_exclusive_group(required=True)
    src.add_argument("--run-dir", type=Path, help="Path to validate_root_profile_scan run dir.")
    src.add_argument("--summary", type=Path, help="Path to summary.json.")
    ap.add_argument("--top", type=int, default=10, help="How many mu points to show.")
    args = ap.parse_args()

    if args.summary:
        run_dir = args.summary.resolve().parent
        summary_path = args.summary.resolve()
    else:
        run_dir = args.run_dir.resolve()
        summary_path = run_dir / "summary.json"

    if not summary_path.exists():
        raise SystemExit(f"Missing: {summary_path}")

    summary = _load_json(summary_path)
    artifacts = summary.get("artifacts", {})

    root_path = Path(artifacts.get("root_profile_scan_json", run_dir / "root_profile_scan.json"))
    ns_path = Path(
        artifacts.get("nextstat_profile_scan_json", run_dir / "nextstat_profile_scan.json")
    )

    if not root_path.is_absolute():
        root_path = (run_dir / root_path).resolve()
    if not ns_path.is_absolute():
        ns_path = (run_dir / ns_path).resolve()

    if not root_path.exists():
        raise SystemExit(f"Missing: {root_path}")
    if not ns_path.exists():
        raise SystemExit(f"Missing: {ns_path}")

    root = _load_json(root_path)
    ns = _load_json(ns_path)

    root_points = root.get("points") or []
    ns_points = (ns.get("points") or [])
    ns_by_mu = {float(p["mu"]): p for p in ns_points}

    diffs: List[Tuple[float, float]] = []
    missing = 0
    for p in root_points:
        mu = float(p["mu"])
        q_root = float(p["q_mu"])
        p_ns = ns_by_mu.get(mu)
        if p_ns is None:
            missing += 1
            continue
        q_ns = float(p_ns["q_mu"])
        diffs.append((mu, q_ns - q_root))

    max_abs = max((abs(dq) for _, dq in diffs), default=float("nan"))
    mu_at_max = None
    if diffs:
        mu_at_max = max(diffs, key=lambda t: abs(t[1]))[0]

    mu_hat_root = summary.get("root", {}).get("mu_hat")
    mu_hat_ns = summary.get("nextstat", {}).get("mu_hat")
    nll_hat_root = summary.get("root", {}).get("nll_hat")
    nll_hat_ns = summary.get("nextstat", {}).get("nll_hat")

    print("=" * 88)
    print("ROOT vs NextStat profile-scan diff")
    print("=" * 88)
    print(f"run_dir: {run_dir}")
    print(f"missing mu matches: {missing}")
    print()

    print("Fit:")
    if mu_hat_root is not None and mu_hat_ns is not None:
        print(f"  mu_hat root={float(mu_hat_root):.12g} nextstat={float(mu_hat_ns):.12g} d={float(mu_hat_ns)-float(mu_hat_root):+.3e}")
    if nll_hat_root is not None and nll_hat_ns is not None:
        print(f"  nll_hat root={float(nll_hat_root):.12g} nextstat={float(nll_hat_ns):.12g} d={float(nll_hat_ns)-float(nll_hat_root):+.3e}")

    print()
    print("q(mu):")
    if math.isfinite(max_abs):
        print(f"  max_abs_delta_q_mu={max_abs:.6g} at mu={mu_at_max}")
    else:
        print("  no comparable points")

    for mu, dq in _topk(diffs, args.top):
        print(f"  mu={mu:>10.6g}  delta_q_mu={dq:+.6g}")

    print()
    print("Timing (s):")
    for k, v in (summary.get("timing_s") or {}).items():
        try:
            print(f"  {k}: {float(v):.6g}")
        except Exception:
            print(f"  {k}: {v}")

    return 0


if __name__ == "__main__":
    raise SystemExit(main())

