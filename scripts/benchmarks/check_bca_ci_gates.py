#!/usr/bin/env python3
"""Check BCa CI benchmark gate thresholds (HEP + churn).

Inputs:
- HEP summary JSON from `bench_unbinned_summary_ci.py`
- churn summary JSON from `bench_churn_bootstrap_ci_methods.py`

Outputs:
- JSON gate report
- optional Markdown gate report

Exits with code 1 if any gate fails.
"""

from __future__ import annotations

import argparse
import json
import sys
from pathlib import Path
from typing import Any


def _load_json(path: Path) -> dict[str, Any]:
    try:
        return json.loads(path.read_text(encoding="utf-8"))
    except FileNotFoundError as exc:
        raise RuntimeError(f"missing file: {path}") from exc
    except json.JSONDecodeError as exc:
        raise RuntimeError(f"invalid JSON in {path}: {exc}") from exc


def _safe_div(num: float, den: float) -> float | None:
    if den <= 0:
        return None
    return num / den


def _fmt_float(value: float | None, suffix: str = "") -> str:
    if value is None:
        return "n/a"
    return f"{value:.3f}{suffix}"


def _build_report(
    hep: dict[str, Any],
    churn: dict[str, Any],
    *,
    hep_max_overhead: float,
    churn_max_overhead: float,
    hep_max_fallback_rate: float,
    churn_max_fallback_rate: float,
    hep_min_effective_bca_rate: float,
    churn_min_effective_bca_rate: float,
) -> dict[str, Any]:
    failures: list[str] = []

    # HEP metrics
    hep_p = hep.get("percentile", {})
    hep_b = hep.get("bca", {})
    hep_runs = int(hep_b.get("runs", 0))
    hep_overhead = _safe_div(float(hep_b.get("median_wall_s", 0.0)), float(hep_p.get("median_wall_s", 0.0)))
    hep_fallback_rate = _safe_div(float(hep_b.get("fallback_count", 0.0)), float(hep_runs))
    hep_effective_bca_rate = _safe_div(float(hep_b.get("effective_bca_count", 0.0)), float(hep_runs))

    # Churn metrics
    churn_p = churn.get("percentile", {})
    churn_b = churn.get("bca", {})
    churn_overhead = _safe_div(
        float(churn_b.get("median_wall_s", 0.0)), float(churn_p.get("median_wall_s", 0.0))
    )
    churn_effective_total = float(churn_b.get("effective_bca_total", 0.0))
    churn_fallback_total = float(churn_b.get("fallback_total", 0.0))
    churn_total_coeff = churn_effective_total + churn_fallback_total
    churn_fallback_rate = _safe_div(churn_fallback_total, churn_total_coeff)
    churn_effective_bca_rate = _safe_div(churn_effective_total, churn_total_coeff)

    # Gate checks
    if hep_overhead is None:
        failures.append("HEP overhead is undefined")
    elif hep_overhead > hep_max_overhead:
        failures.append(
            f"HEP overhead {hep_overhead:.3f}x exceeds threshold {hep_max_overhead:.3f}x"
        )
    if hep_fallback_rate is None:
        failures.append("HEP fallback rate is undefined")
    elif hep_fallback_rate > hep_max_fallback_rate:
        failures.append(
            f"HEP fallback rate {hep_fallback_rate:.3f} exceeds threshold {hep_max_fallback_rate:.3f}"
        )
    if hep_effective_bca_rate is None:
        failures.append("HEP effective BCa rate is undefined")
    elif hep_effective_bca_rate < hep_min_effective_bca_rate:
        failures.append(
            f"HEP effective BCa rate {hep_effective_bca_rate:.3f} below threshold {hep_min_effective_bca_rate:.3f}"
        )

    if churn_overhead is None:
        failures.append("Churn overhead is undefined")
    elif churn_overhead > churn_max_overhead:
        failures.append(
            f"Churn overhead {churn_overhead:.3f}x exceeds threshold {churn_max_overhead:.3f}x"
        )
    if churn_fallback_rate is None:
        failures.append("Churn fallback rate is undefined")
    elif churn_fallback_rate > churn_max_fallback_rate:
        failures.append(
            f"Churn fallback rate {churn_fallback_rate:.3f} exceeds threshold {churn_max_fallback_rate:.3f}"
        )
    if churn_effective_bca_rate is None:
        failures.append("Churn effective BCa rate is undefined")
    elif churn_effective_bca_rate < churn_min_effective_bca_rate:
        failures.append(
            f"Churn effective BCa rate {churn_effective_bca_rate:.3f} below threshold {churn_min_effective_bca_rate:.3f}"
        )

    report = {
        "schema_version": "bca_ci_gate_v1",
        "overall_pass": len(failures) == 0,
        "thresholds": {
            "hep_max_overhead": hep_max_overhead,
            "churn_max_overhead": churn_max_overhead,
            "hep_max_fallback_rate": hep_max_fallback_rate,
            "churn_max_fallback_rate": churn_max_fallback_rate,
            "hep_min_effective_bca_rate": hep_min_effective_bca_rate,
            "churn_min_effective_bca_rate": churn_min_effective_bca_rate,
        },
        "metrics": {
            "hep": {
                "runs": hep_runs,
                "median_wall_percentile_s": hep_p.get("median_wall_s"),
                "median_wall_bca_s": hep_b.get("median_wall_s"),
                "overhead_bca_vs_percentile": hep_overhead,
                "fallback_count": hep_b.get("fallback_count"),
                "fallback_rate": hep_fallback_rate,
                "effective_bca_count": hep_b.get("effective_bca_count"),
                "effective_bca_rate": hep_effective_bca_rate,
            },
            "churn": {
                "median_wall_percentile_s": churn_p.get("median_wall_s"),
                "median_wall_bca_s": churn_b.get("median_wall_s"),
                "overhead_bca_vs_percentile": churn_overhead,
                "fallback_total": churn_b.get("fallback_total"),
                "fallback_rate": churn_fallback_rate,
                "effective_bca_total": churn_b.get("effective_bca_total"),
                "effective_bca_rate": churn_effective_bca_rate,
            },
        },
        "failures": failures,
    }
    return report


def _write_markdown(report: dict[str, Any], out_md: Path) -> None:
    m = report["metrics"]
    hep = m["hep"]
    churn = m["churn"]
    thr = report["thresholds"]

    lines = [
        "# BCa CI Gate Report",
        "",
        f"- Overall pass: `{str(report['overall_pass']).lower()}`",
        "",
        "## Thresholds",
        "",
        f"- HEP overhead <= `{thr['hep_max_overhead']:.3f}x`",
        f"- Churn overhead <= `{thr['churn_max_overhead']:.3f}x`",
        f"- HEP fallback rate <= `{thr['hep_max_fallback_rate']:.3f}`",
        f"- Churn fallback rate <= `{thr['churn_max_fallback_rate']:.3f}`",
        f"- HEP effective BCa rate >= `{thr['hep_min_effective_bca_rate']:.3f}`",
        f"- Churn effective BCa rate >= `{thr['churn_min_effective_bca_rate']:.3f}`",
        "",
        "## Metrics",
        "",
        "| Workflow | Overhead (bca/percentile) | Fallback rate | Effective BCa rate |",
        "|---|---:|---:|---:|",
        (
            f"| HEP | {_fmt_float(hep['overhead_bca_vs_percentile'], 'x')} | "
            f"{_fmt_float(hep['fallback_rate'])} | {_fmt_float(hep['effective_bca_rate'])} |"
        ),
        (
            f"| Churn | {_fmt_float(churn['overhead_bca_vs_percentile'], 'x')} | "
            f"{_fmt_float(churn['fallback_rate'])} | {_fmt_float(churn['effective_bca_rate'])} |"
        ),
        "",
    ]

    failures = report.get("failures", [])
    if failures:
        lines.append("## Failures")
        lines.append("")
        for f in failures:
            lines.append(f"- {f}")
        lines.append("")

    out_md.write_text("\n".join(lines), encoding="utf-8")


def main() -> None:
    ap = argparse.ArgumentParser(description="Check BCa benchmark gate thresholds")
    ap.add_argument("--hep-summary", required=True, type=Path)
    ap.add_argument("--churn-summary", required=True, type=Path)
    ap.add_argument("--out-json", required=True, type=Path)
    ap.add_argument("--out-md", type=Path)
    ap.add_argument("--hep-max-overhead", type=float, default=1.25)
    ap.add_argument("--churn-max-overhead", type=float, default=1.75)
    ap.add_argument("--hep-max-fallback-rate", type=float, default=0.05)
    ap.add_argument("--churn-max-fallback-rate", type=float, default=0.05)
    ap.add_argument("--hep-min-effective-bca-rate", type=float, default=0.95)
    ap.add_argument("--churn-min-effective-bca-rate", type=float, default=0.95)
    args = ap.parse_args()

    hep = _load_json(args.hep_summary)
    churn = _load_json(args.churn_summary)
    report = _build_report(
        hep,
        churn,
        hep_max_overhead=args.hep_max_overhead,
        churn_max_overhead=args.churn_max_overhead,
        hep_max_fallback_rate=args.hep_max_fallback_rate,
        churn_max_fallback_rate=args.churn_max_fallback_rate,
        hep_min_effective_bca_rate=args.hep_min_effective_bca_rate,
        churn_min_effective_bca_rate=args.churn_min_effective_bca_rate,
    )

    args.out_json.parent.mkdir(parents=True, exist_ok=True)
    args.out_json.write_text(json.dumps(report, indent=2), encoding="utf-8")
    if args.out_md is not None:
        args.out_md.parent.mkdir(parents=True, exist_ok=True)
        _write_markdown(report, args.out_md)

    if report["overall_pass"]:
        print("BCa CI gates: PASS")
        sys.exit(0)

    print("BCa CI gates: FAIL")
    for f in report["failures"]:
        print(f"- {f}")
    sys.exit(1)


if __name__ == "__main__":
    main()
