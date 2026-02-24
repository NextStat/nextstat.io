#!/usr/bin/env python3
"""Render a human README snippet for the Bayesian suite with ranking."""

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
    ap.add_argument("--suite", required=True, help="Path to bayesian_suite.json")
    ap.add_argument("--out", required=True, help="Output Markdown path")
    args = ap.parse_args()

    suite_path = Path(args.suite).resolve()
    out_path = Path(args.out).resolve()
    out_path.parent.mkdir(parents=True, exist_ok=True)

    obj = json.loads(suite_path.read_text())
    cases = obj.get("cases") if isinstance(obj.get("cases"), list) else []

    lines: list[str] = []
    lines.append("# Bayesian suite (NUTS diagnostics + ESS/sec)")
    lines.append("")
    lines.append("This snapshot reports sampler health metrics (rank-normalized R-hat, Geyer ESS, E-BFMI) and a simple ESS/sec proxy computed as `min(ESS_bulk)/wall_time` (includes warmup).")
    lines.append("")

    # -- Detailed results table --
    lines.append("## Detailed results")
    lines.append("")
    lines.append("| Case | Backend | Status | Wall (s) | min ESS_bulk | min ESS_tail | max R-hat | ESS/grad | min ESS_bulk/s |")
    lines.append("|---|---|---|---:|---:|---:|---:|---:|---:|")
    for c in cases:
        status = str(c.get("status") or "unknown")
        lines.append(
            "| "
            + " | ".join(
                [
                    str(c.get("case") or "unknown"),
                    str(c.get("backend") or "unknown"),
                    status,
                    _fmt(c.get("wall_time_s")),
                    _fmt(c.get("min_ess_bulk")),
                    _fmt(c.get("min_ess_tail")),
                    _fmt(c.get("max_r_hat")),
                    _fmt(c.get("ess_per_grad"), digits=4),
                    _fmt(c.get("min_ess_bulk_per_sec")),
                ]
            )
            + " |"
        )
    lines.append("")

    # -- ESS/sec decomposition table --
    decomp_rows = [c for c in cases if c.get("status") == "ok" and _safe_float(c.get("ess_per_grad")) is not None]
    if decomp_rows:
        lines.append("## ESS/sec Decomposition")
        lines.append("")
        lines.append("`ESS/sec = (ESS/grad) × (grad/sec)`")
        lines.append("")
        lines.append("| Case | Backend | ESS/grad | grad/s | ESS_bulk/s | Product check |")
        lines.append("|---|---|---:|---:|---:|---:|")
        for c in decomp_rows:
            epg = _safe_float(c.get("ess_per_grad"))
            gps = _safe_float(c.get("grad_per_sec"))
            eps = _safe_float(c.get("min_ess_bulk_per_sec"))
            prod = (epg * gps) if (epg is not None and gps is not None) else None
            lines.append(
                "| "
                + " | ".join(
                    [
                        str(c.get("case") or "unknown"),
                        str(c.get("backend") or "unknown"),
                        _fmt(epg, digits=4),
                        _fmt(gps),
                        _fmt(eps),
                        _fmt(prod),
                    ]
                )
                + " |"
            )
        lines.append("")

    # -- Posterior parity (NextStat dense vs diagonal) --
    parity = obj.get("parity") if isinstance(obj.get("parity"), dict) else {}
    parity_rows = parity.get("rows") if isinstance(parity.get("rows"), list) else []
    if parity_rows:
        lines.append("## Posterior parity: dense vs diagonal (mean z-scores)")
        lines.append("")
        lines.append(f"Thresholds: warn z ≥ `{parity.get('warn_z', '—')}`, fail z ≥ `{parity.get('fail_z', '—')}`")
        lines.append("")
        lines.append("| Case | Status | max z | Worst params (z) |")
        lines.append("|---|---|---:|---|")
        for r in parity_rows:
            worst = r.get("worst") if isinstance(r.get("worst"), list) else []
            worst_s = ", ".join(
                f"{w.get('param','?')}({_fmt(w.get('z'), digits=2)})" for w in worst if isinstance(w, dict)
            ) or "—"
            lines.append(
                "| {case} | {status} | {maxz} | {worst} |".format(
                    case=str(r.get("case") or "unknown"),
                    status=str(r.get("status") or "unknown"),
                    maxz=_fmt(r.get("max_z"), digits=2),
                    worst=worst_s,
                )
            )
        lines.append("")

    # -- Per-case ranking table --
    # Group by case, rank backends by ESS_bulk/sec
    by_case: dict[str, list[dict]] = defaultdict(list)
    for c in cases:
        by_case[str(c.get("case") or "unknown")].append(c)

    lines.append("## Ranking by ESS_bulk/sec (per case)")
    lines.append("")
    lines.append("For each case, backends are ranked by `min ESS_bulk/sec`. The fastest backend is the baseline (1.00x).")
    lines.append("")
    lines.append("| Case | Rank | Backend | ESS_bulk/s | Speedup | Note |")
    lines.append("|---|---:|---|---:|---:|---|")

    for case_id, entries in by_case.items():
        scored: list[tuple[float, dict]] = []
        skipped: list[dict] = []
        for e in entries:
            v = _safe_float(e.get("min_ess_bulk_per_sec"))
            status = str(e.get("status") or "unknown")
            if v is not None and v > 0 and status == "ok":
                scored.append((v, e))
            else:
                skipped.append(e)

        scored.sort(key=lambda t: t[0], reverse=True)
        best = scored[0][0] if scored else 1.0

        for rank, (val, e) in enumerate(scored, 1):
            speedup = val / best if best > 0 else 0.0
            note = ""
            if rank == 1:
                note = "**winner**"
            elif rank == len(scored):
                note = "slowest"
            lines.append(
                f"| {case_id} | {rank} | {e.get('backend', '?')} | {_fmt(val)} | {speedup:.2f}x | {note} |"
            )
        for e in skipped:
            status = str(e.get("status") or "unknown")
            reason = "not supported" if status == "warn" else status
            lines.append(
                f"| {case_id} | — | {e.get('backend', '?')} | — | — | {reason} |"
            )

    lines.append("")

    # -- Overall summary --
    backend_scores: dict[str, list[float]] = defaultdict(list)
    for c in cases:
        v = _safe_float(c.get("min_ess_bulk_per_sec"))
        status = str(c.get("status") or "unknown")
        if v is not None and v > 0 and status == "ok":
            backend_scores[str(c.get("backend") or "unknown")].append(v)

    if backend_scores:
        lines.append("## Overall summary (geometric mean ESS_bulk/sec across cases)")
        lines.append("")
        lines.append("| Backend | Cases | Geomean ESS_bulk/s |")
        lines.append("|---|---:|---:|")
        gmeans: list[tuple[float, str]] = []
        for backend, vals in sorted(backend_scores.items()):
            if vals:
                log_mean = sum(math.log(v) for v in vals) / len(vals)
                gmean = math.exp(log_mean)
                gmeans.append((gmean, backend))
                lines.append(f"| {backend} | {len(vals)} | {_fmt(gmean)} |")
        lines.append("")

        if gmeans:
            gmeans.sort(reverse=True)
            lines.append(f"**Overall winner**: {gmeans[0][1]} ({_fmt(gmeans[0][0])} ESS_bulk/sec geomean)")
            lines.append("")

    meta = obj.get("meta", {})
    lines.append("---")
    lines.append(f"*Generated by nextstat {meta.get('nextstat_version', '?')}, Python {meta.get('python', '?')}, {meta.get('platform', '?')}*")
    lines.append("")

    out_path.write_text("\n".join(lines) + "\n")
    return 0


if __name__ == "__main__":
    raise SystemExit(main())
