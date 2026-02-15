#!/usr/bin/env python3
"""Render Markdown report from MAMS benchmark suite results."""

from __future__ import annotations

import argparse
import json
import math
from collections import defaultdict
from pathlib import Path


def _fmt(x, *, digits: int = 3) -> str:
    try:
        v = float(x)
    except Exception:
        return "—"
    if v != v:
        return "—"
    if abs(v) >= 1000:
        return f"{v:,.0f}"
    if abs(v) >= 100:
        return f"{v:.0f}"
    if abs(v) >= 10:
        return f"{v:.1f}"
    return f"{v:.{digits}f}".rstrip("0").rstrip(".")


def _safe_float(x) -> float | None:
    try:
        v = float(x)
    except Exception:
        return None
    if not math.isfinite(v):
        return None
    return v


def main() -> int:
    ap = argparse.ArgumentParser()
    ap.add_argument("suite_dir", help="Path to suite results directory (contains mams_suite.json)")
    args = ap.parse_args()

    suite_dir = Path(args.suite_dir).resolve()
    suite_path = suite_dir / "mams_suite.json"
    if not suite_path.exists():
        print(f"ERROR: {suite_path} not found", file=__import__("sys").stderr)
        return 1

    obj = json.loads(suite_path.read_text())
    cases = obj.get("cases", [])
    meta = obj.get("meta", {})
    config = obj.get("config", {})

    lines: list[str] = []
    lines.append("# MAMS Benchmark Suite Results")
    lines.append("")
    lines.append(f"Config: {config.get('n_chains', '?')} chains, "
                 f"warmup={config.get('n_warmup', '?')}, "
                 f"samples={config.get('n_samples', '?')}, "
                 f"target_accept={config.get('target_accept', '?')}")
    lines.append("")

    # -- Aggregate across seeds: take median for each (case, backend) --
    grouped: dict[tuple[str, str], list[dict]] = defaultdict(list)
    for c in cases:
        key = (str(c.get("case", "?")), str(c.get("backend", "?")))
        grouped[key].append(c)

    def median_metric(entries: list[dict], key: str) -> float | None:
        vals = [v for v in (_safe_float(e.get(key)) for e in entries) if v is not None]
        if not vals:
            return None
        vals.sort()
        n = len(vals)
        mid = n // 2
        return vals[mid] if n % 2 else 0.5 * (vals[mid - 1] + vals[mid])

    def range_metric(entries: list[dict], key: str) -> tuple[float | None, float | None]:
        vals = [v for v in (_safe_float(e.get(key)) for e in entries) if v is not None]
        if not vals:
            return None, None
        return min(vals), max(vals)

    # -- Detailed table --
    lines.append("## Detailed Results (median across seeds)")
    lines.append("")
    lines.append("| Case | Backend | ESS/grad | [min–max] | Seeds | grad/s | ESS/s | Wall (s) | min ESS_bulk | R-hat | Accept |")
    lines.append("|---|---|---:|:---:|:---:|---:|---:|---:|---:|---:|---:|")

    agg_rows: list[dict] = []
    for (case, backend), entries in sorted(grouped.items()):
        ok_entries = [e for e in entries if e.get("status") == "ok"]
        n_seeds = len(ok_entries)
        if not ok_entries:
            lines.append(f"| {case} | {backend} | — | — | 0 | — | — | — | — | — |")
            continue

        row = {
            "case": case,
            "backend": backend,
            "ess_per_grad": median_metric(ok_entries, "ess_per_grad"),
            "ess_per_sec": median_metric(ok_entries, "ess_per_sec"),
            "wall_time_s": median_metric(ok_entries, "wall_time_s"),
            "n_grad_evals": median_metric(ok_entries, "n_grad_evals"),
            "min_ess_bulk": median_metric(ok_entries, "min_ess_bulk"),
            "max_r_hat": median_metric(ok_entries, "max_r_hat"),
        }
        # Decomposition metric: grad/s = n_grad_evals / wall_time_s (per run, then median).
        grad_per_sec_vals = []
        for e in ok_entries:
            n_grad = _safe_float(e.get("n_grad_evals"))
            wall = _safe_float(e.get("wall_time_s"))
            if n_grad is not None and wall is not None and wall > 0.0:
                grad_per_sec_vals.append(n_grad / wall)
        if grad_per_sec_vals:
            grad_per_sec_vals.sort()
            mid = len(grad_per_sec_vals) // 2
            row["grad_per_sec"] = (
                grad_per_sec_vals[mid]
                if len(grad_per_sec_vals) % 2
                else 0.5 * (grad_per_sec_vals[mid - 1] + grad_per_sec_vals[mid])
            )
        else:
            row["grad_per_sec"] = None
        # Accept rate from individual run objects (need to load)
        accept_vals = []
        for e in ok_entries:
            p = suite_dir / e.get("path", "")
            if p.exists():
                try:
                    r = json.loads(p.read_text())
                    ar = _safe_float(r.get("metrics", {}).get("accept_rate"))
                    if ar is not None:
                        accept_vals.append(ar)
                except Exception:
                    pass
        row["accept_rate"] = (sorted(accept_vals)[len(accept_vals) // 2]
                              if accept_vals else None)
        epg_lo, epg_hi = range_metric(ok_entries, "ess_per_grad")
        row["epg_range"] = (epg_lo, epg_hi)
        agg_rows.append(row)

        rng_str = "—"
        if epg_lo is not None and epg_hi is not None:
            rng_str = f"{_fmt(epg_lo, digits=4)}–{_fmt(epg_hi, digits=4)}"

        lines.append(
            f"| {case} | {backend} "
            f"| {_fmt(row['ess_per_grad'], digits=4)} "
            f"| {rng_str} "
            f"| {n_seeds} "
            f"| {_fmt(row['grad_per_sec'])} "
            f"| {_fmt(row['ess_per_sec'])} "
            f"| {_fmt(row['wall_time_s'])} "
            f"| {_fmt(row['min_ess_bulk'])} "
            f"| {_fmt(row['max_r_hat'])} "
            f"| {_fmt(row['accept_rate'])} |"
        )

    lines.append("")

    # -- ESS/sec decomposition table --
    lines.append("## ESS/sec Decomposition")
    lines.append("")
    lines.append("`ESS/sec = (ESS/grad) × (grad/sec)`")
    lines.append("")
    lines.append("| Case | Backend | ESS/grad | grad/s | ESS/s | Product check |")
    lines.append("|---|---|---:|---:|---:|---:|")
    for r in sorted(agg_rows, key=lambda x: (x["case"], x["backend"])):
        epg = _safe_float(r.get("ess_per_grad"))
        gps = _safe_float(r.get("grad_per_sec"))
        eps = _safe_float(r.get("ess_per_sec"))
        prod = (epg * gps) if (epg is not None and gps is not None) else None
        lines.append(
            f"| {r['case']} | {r['backend']} | {_fmt(epg, digits=4)} | {_fmt(gps)} | {_fmt(eps)} | {_fmt(prod)} |"
        )
    lines.append("")

    # -- Speedup table (MAMS vs NUTS baseline) --
    lines.append("## MAMS vs NUTS Speedup (ESS/gradient)")
    lines.append("")
    lines.append("| Case | MAMS ESS/grad | NUTS ESS/grad | Ratio (MAMS/NUTS) |")
    lines.append("|---|---:|---:|---:|")

    by_case: dict[str, dict[str, dict]] = defaultdict(dict)
    for r in agg_rows:
        by_case[r["case"]][r["backend"]] = r

    for case in sorted(by_case):
        mams = by_case[case].get("nextstat_mams", {})
        nuts = by_case[case].get("nextstat_nuts", {})
        mams_epg = _safe_float(mams.get("ess_per_grad"))
        nuts_epg = _safe_float(nuts.get("ess_per_grad"))
        ratio = ""
        if mams_epg and nuts_epg and nuts_epg > 0:
            ratio = f"{mams_epg / nuts_epg:.2f}x"
        lines.append(
            f"| {case} | {_fmt(mams_epg, digits=4)} | {_fmt(nuts_epg, digits=4)} | {ratio} |"
        )

    lines.append("")

    # -- Posterior parity (MAMS vs NUTS) --
    parity = obj.get("parity") if isinstance(obj.get("parity"), dict) else {}
    parity_rows = parity.get("rows") if isinstance(parity.get("rows"), list) else []
    if parity_rows:
        lines.append("## Posterior Parity: MAMS vs NUTS (mean z-scores)")
        lines.append("")
        lines.append(f"Thresholds: warn z ≥ `{parity.get('warn_z', '—')}`, fail z ≥ `{parity.get('fail_z', '—')}`")
        lines.append("")
        lines.append("| Case | Seed | Status | max z | Worst params (z) |")
        lines.append("|---|---:|---|---:|---|")
        for r in parity_rows:
            worst = r.get("worst") if isinstance(r.get("worst"), list) else []
            worst_s = ", ".join(
                f"{w.get('param','?')}({_fmt(w.get('z'), digits=2)})" for w in worst if isinstance(w, dict)
            ) or "—"
            lines.append(
                f"| {r.get('case','?')} | {r.get('seed','?')} | {r.get('status','?')} | {_fmt(r.get('max_z'), digits=2)} | {worst_s} |"
            )
        lines.append("")

    lines.append("---")
    lines.append(f"*Generated by nextstat {meta.get('nextstat_version', '?')}, "
                 f"Python {meta.get('python', '?')}, {meta.get('platform', '?')}*")
    lines.append("")

    report_text = "\n".join(lines)
    print(report_text)

    report_path = suite_dir / "mams_benchmark_report.md"
    report_path.write_text(report_text)

    return 0


if __name__ == "__main__":
    raise SystemExit(main())
