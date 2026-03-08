#!/usr/bin/env python3
"""Render detailed and published-snapshot Markdown reports for the MAMS suite."""

from __future__ import annotations

import argparse
import json
import math
import sys
from collections import defaultdict
from pathlib import Path
from typing import Any


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
    if abs(v) >= 1:
        return f"{v:.{digits}f}".rstrip("0").rstrip(".")
    if v == 0:
        return "0"
    return f"{v:.{digits}g}"


def _safe_float(x) -> float | None:
    try:
        v = float(x)
    except Exception:
        return None
    if not math.isfinite(v):
        return None
    return v


def _load_json(path: Path) -> dict[str, Any]:
    return json.loads(path.read_text(encoding="utf-8"))


def _resolve_suite_path(args: argparse.Namespace) -> tuple[Path, Path]:
    if str(args.suite).strip():
        suite_path = Path(args.suite).resolve()
        suite_dir = suite_path.parent
    elif args.suite_dir:
        suite_dir = Path(args.suite_dir).resolve()
        suite_path = suite_dir / "mams_suite.json"
    else:
        raise SystemExit("provide either suite_dir or --suite")
    if not suite_path.exists():
        raise SystemExit(f"missing suite artifact: {suite_path}")
    return suite_dir, suite_path


def _median_metric(entries: list[dict[str, Any]], key: str) -> float | None:
    vals = [v for v in (_safe_float(e.get(key)) for e in entries) if v is not None]
    if not vals:
        return None
    vals.sort()
    n = len(vals)
    mid = n // 2
    return vals[mid] if n % 2 else 0.5 * (vals[mid - 1] + vals[mid])


def _range_metric(entries: list[dict[str, Any]], key: str) -> tuple[float | None, float | None]:
    vals = [v for v in (_safe_float(e.get(key)) for e in entries) if v is not None]
    if not vals:
        return None, None
    return min(vals), max(vals)


def _aggregate_rows(suite_dir: Path, cases: list[dict[str, Any]]) -> list[dict[str, Any]]:
    grouped: dict[tuple[str, str], list[dict[str, Any]]] = defaultdict(list)
    for case in cases:
        key = (str(case.get("case", "?")), str(case.get("backend", "?")))
        grouped[key].append(case)

    agg_rows: list[dict[str, Any]] = []
    for (case, backend), entries in sorted(grouped.items()):
        ok_entries = [e for e in entries if e.get("status") == "ok"]
        row = {
            "case": case,
            "backend": backend,
            "status": "ok" if ok_entries else "warn",
            "n_seeds": len(ok_entries),
            "ess_per_grad": _median_metric(ok_entries, "ess_per_grad"),
            "ess_per_sec": _median_metric(ok_entries, "ess_per_sec"),
            "wall_time_s": _median_metric(ok_entries, "wall_time_s"),
            "n_grad_evals": _median_metric(ok_entries, "n_grad_evals"),
            "min_ess_bulk": _median_metric(ok_entries, "min_ess_bulk"),
            "max_r_hat": _median_metric(ok_entries, "max_r_hat"),
        }
        grad_per_sec_vals = []
        for entry in ok_entries:
            n_grad = _safe_float(entry.get("n_grad_evals"))
            wall = _safe_float(entry.get("wall_time_s"))
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

        accept_vals = []
        for entry in ok_entries:
            path = suite_dir / entry.get("path", "")
            if not path.exists():
                continue
            try:
                run_doc = _load_json(path)
            except Exception:
                continue
            accept_rate = _safe_float(run_doc.get("metrics", {}).get("accept_rate"))
            if accept_rate is not None:
                accept_vals.append(accept_rate)
        row["accept_rate"] = (
            sorted(accept_vals)[len(accept_vals) // 2]
            if accept_vals
            else None
        )
        row["epg_range"] = _range_metric(ok_entries, "ess_per_grad")
        agg_rows.append(row)
    return agg_rows


def _render_detailed_report(suite_dir: Path, suite_obj: dict[str, Any]) -> str:
    cases = suite_obj.get("cases") if isinstance(suite_obj.get("cases"), list) else []
    meta = suite_obj.get("meta", {}) if isinstance(suite_obj.get("meta"), dict) else {}
    config = suite_obj.get("config", {}) if isinstance(suite_obj.get("config"), dict) else {}
    agg_rows = _aggregate_rows(suite_dir, [c for c in cases if isinstance(c, dict)])

    lines: list[str] = []
    lines.append("# MAMS Benchmark Suite Results")
    lines.append("")
    lines.append(
        f"Config: {config.get('n_chains', '?')} chains, "
        f"warmup={config.get('n_warmup', '?')}, "
        f"samples={config.get('n_samples', '?')}, "
        f"target_accept={config.get('target_accept', '?')}"
    )
    lines.append("")
    lines.append("## Detailed Results (median across seeds)")
    lines.append("")
    lines.append("| Case | Backend | ESS/grad | [min–max] | Seeds | grad/s | ESS/s | Wall (s) | min ESS_bulk | R-hat | Accept |")
    lines.append("|---|---|---:|:---:|:---:|---:|---:|---:|---:|---:|---:|")

    for row in agg_rows:
        if row["n_seeds"] == 0:
            lines.append(f"| {row['case']} | {row['backend']} | — | — | 0 | — | — | — | — | — | — |")
            continue
        epg_lo, epg_hi = row["epg_range"]
        rng_str = "—"
        if epg_lo is not None and epg_hi is not None:
            rng_str = f"{_fmt(epg_lo, digits=4)}–{_fmt(epg_hi, digits=4)}"
        lines.append(
            f"| {row['case']} | {row['backend']} "
            f"| {_fmt(row['ess_per_grad'], digits=4)} "
            f"| {rng_str} "
            f"| {row['n_seeds']} "
            f"| {_fmt(row['grad_per_sec'])} "
            f"| {_fmt(row['ess_per_sec'])} "
            f"| {_fmt(row['wall_time_s'])} "
            f"| {_fmt(row['min_ess_bulk'])} "
            f"| {_fmt(row['max_r_hat'])} "
            f"| {_fmt(row['accept_rate'])} |"
        )

    lines.append("")
    lines.append("## ESS/sec Decomposition")
    lines.append("")
    lines.append("`ESS/sec = (ESS/grad) × (grad/sec)`")
    lines.append("")
    lines.append("| Case | Backend | ESS/grad | grad/s | ESS/s | Product check |")
    lines.append("|---|---|---:|---:|---:|---:|")
    for row in agg_rows:
        epg = _safe_float(row.get("ess_per_grad"))
        gps = _safe_float(row.get("grad_per_sec"))
        eps = _safe_float(row.get("ess_per_sec"))
        prod = (epg * gps) if (epg is not None and gps is not None) else None
        lines.append(
            f"| {row['case']} | {row['backend']} | {_fmt(epg, digits=4)} | {_fmt(gps)} | {_fmt(eps)} | {_fmt(prod)} |"
        )
    lines.append("")

    lines.append("## MAMS vs NUTS Speedup (ESS/gradient)")
    lines.append("")
    lines.append("| Case | MAMS ESS/grad | NUTS ESS/grad | Ratio (MAMS/NUTS) |")
    lines.append("|---|---:|---:|---:|")
    by_case: dict[str, dict[str, dict[str, Any]]] = defaultdict(dict)
    for row in agg_rows:
        by_case[row["case"]][row["backend"]] = row
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

    parity = suite_obj.get("parity") if isinstance(suite_obj.get("parity"), dict) else {}
    parity_rows = parity.get("rows") if isinstance(parity.get("rows"), list) else []
    if parity_rows:
        lines.append("## Posterior Parity: MAMS vs NUTS (mean z-scores)")
        lines.append("")
        lines.append(
            f"Thresholds: warn z ≥ `{parity.get('warn_z', '—')}`, fail z ≥ `{parity.get('fail_z', '—')}`"
        )
        lines.append("")
        lines.append("| Case | Seed | Status | max z | Worst params (z) |")
        lines.append("|---|---:|---|---:|---|")
        for row in parity_rows:
            worst = row.get("worst") if isinstance(row.get("worst"), list) else []
            worst_s = ", ".join(
                f"{w.get('param','?')}({_fmt(w.get('z'), digits=2)})" for w in worst if isinstance(w, dict)
            ) or "—"
            lines.append(
                f"| {row.get('case','?')} | {row.get('seed','?')} | {row.get('status','?')} | {_fmt(row.get('max_z'), digits=2)} | {worst_s} |"
            )
        lines.append("")

    lines.append("---")
    lines.append(
        f"*Generated by nextstat {meta.get('nextstat_version', '?')}, "
        f"Python {meta.get('python', '?')}, {meta.get('platform', '?')}*"
    )
    lines.append("")
    return "\n".join(lines)


def _render_snapshot_snippet(suite_dir: Path, suite_obj: dict[str, Any], assessment: dict[str, Any] | None) -> str:
    cases = suite_obj.get("cases") if isinstance(suite_obj.get("cases"), list) else []
    config = suite_obj.get("config", {}) if isinstance(suite_obj.get("config"), dict) else {}
    assessment = assessment if isinstance(assessment, dict) else {}
    promotion_gate = assessment.get("promotion_gate") if isinstance(assessment.get("promotion_gate"), dict) else {}
    core_quality = assessment.get("core_quality") if isinstance(assessment.get("core_quality"), dict) else {}
    review_summary = promotion_gate.get("review_summary") if isinstance(promotion_gate.get("review_summary"), dict) else {}
    policy = promotion_gate.get("policy") if isinstance(promotion_gate.get("policy"), dict) else {}
    agg_rows = _aggregate_rows(suite_dir, [c for c in cases if isinstance(c, dict)])

    lines: list[str] = []
    lines.append("# MAMS suite (stable-surface diagnostics + efficiency)")
    lines.append("")
    lines.append(
        "This snapshot reports the tracked CPU `nextstat_mams` vs `nextstat_nuts` suite and keeps stable-surface policy separate from raw benchmark throughput."
    )
    lines.append(
        f"Canonical config: `{config.get('n_chains', '?')} chains`, `warmup={config.get('n_warmup', '?')}`, `samples={config.get('n_samples', '?')}`, `target_accept={config.get('target_accept', '?')}`."
    )
    lines.append("")

    if assessment:
        lines.append("## Health verdict")
        lines.append("")
        lines.append(f"- core quality: `{core_quality.get('status', 'unknown')}`")
        lines.append(
            f"- promotion gate ({promotion_gate.get('target_backend', 'unknown')}): `{promotion_gate.get('status', 'unknown')}`"
        )
        failing_cases = review_summary.get("failing_cases") if isinstance(review_summary.get("failing_cases"), list) else []
        lines.append(f"- reviewed cases: `{review_summary.get('n_reviewed_cases', 0)}`")
        lines.append(f"- failing cases: `{', '.join(str(x) for x in failing_cases) if failing_cases else '—'}`")
        lines.append("")
        lines.append("| Metric | Worst case | Observed | Policy |")
        lines.append("|---|---|---:|---:|")
        for label, summary_key, policy_key in [
            ("max_r_hat", "worst_max_r_hat", "max_r_hat"),
            ("min_ess_bulk", "worst_min_ess_bulk", "min_ess_bulk"),
            ("ess_per_sec", "worst_ess_per_sec", "min_ess_per_sec"),
        ]:
            metric_summary = review_summary.get(summary_key) if isinstance(review_summary.get(summary_key), dict) else {}
            lines.append(
                "| {label} | {case} | {observed} | {policy} |".format(
                    label=label,
                    case=str(metric_summary.get("case") or "—"),
                    observed=_fmt(metric_summary.get("value"), digits=4),
                    policy=_fmt(policy.get(policy_key), digits=4),
                )
            )
        failures = promotion_gate.get("failures") if isinstance(promotion_gate.get("failures"), list) else []
        if failures:
            lines.append("")
            lines.append("### Promotion failures")
            lines.append("")
            for row in failures:
                if not isinstance(row, dict):
                    continue
                case_label = row.get("case") or "global"
                if "observed" in row and "threshold" in row:
                    lines.append(
                        f"- `{case_label}`: `{row.get('reason', 'unknown')}` (observed `{_fmt(row.get('observed'), digits=4)}`, threshold `{_fmt(row.get('threshold'), digits=4)}`)"
                    )
                else:
                    lines.append(f"- `{case_label}`: `{row.get('reason', 'unknown')}`")
            lines.append("")

    lines.append("## Detailed results")
    lines.append("")
    lines.append("| Case | Backend | Status | Wall (s) | min ESS_bulk | max R-hat | ESS/grad | ESS/s |")
    lines.append("|---|---|---|---:|---:|---:|---:|---:|")
    for row in agg_rows:
        lines.append(
            f"| {row['case']} | {row['backend']} | {row['status']} | {_fmt(row['wall_time_s'])} | {_fmt(row['min_ess_bulk'])} | {_fmt(row['max_r_hat'], digits=4)} | {_fmt(row['ess_per_grad'], digits=4)} | {_fmt(row['ess_per_sec'])} |"
        )
    lines.append("")

    lines.append("## MAMS vs NUTS speedup (ESS/gradient)")
    lines.append("")
    lines.append("| Case | MAMS ESS/grad | NUTS ESS/grad | Ratio (MAMS/NUTS) |")
    lines.append("|---|---:|---:|---:|")
    by_case: dict[str, dict[str, dict[str, Any]]] = defaultdict(dict)
    for row in agg_rows:
        by_case[row["case"]][row["backend"]] = row
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

    parity = suite_obj.get("parity") if isinstance(suite_obj.get("parity"), dict) else {}
    parity_rows = parity.get("rows") if isinstance(parity.get("rows"), list) else []
    if parity_rows:
        lines.append("## Posterior parity: MAMS vs NUTS (mean z-scores)")
        lines.append("")
        lines.append(
            f"Thresholds: warn z ≥ `{parity.get('warn_z', '—')}`, fail z ≥ `{parity.get('fail_z', '—')}`"
        )
        lines.append("")
        lines.append("| Case | Status | max z | Worst params (z) |")
        lines.append("|---|---|---:|---|")
        for row in parity_rows:
            worst = row.get("worst") if isinstance(row.get("worst"), list) else []
            worst_s = ", ".join(
                f"{w.get('param','?')}({_fmt(w.get('z'), digits=2)})" for w in worst if isinstance(w, dict)
            ) or "—"
            lines.append(
                "| {case} | {status} | {maxz} | {worst} |".format(
                    case=str(row.get("case") or "unknown"),
                    status=str(row.get("status") or "unknown"),
                    maxz=_fmt(row.get("max_z"), digits=2),
                    worst=worst_s,
                )
            )
        lines.append("")

    return "\n".join(lines)


def main() -> int:
    ap = argparse.ArgumentParser()
    ap.add_argument("suite_dir", nargs="?", help="Path to suite results directory (contains mams_suite.json)")
    ap.add_argument("--suite", default="", help="Optional path to mams_suite.json")
    ap.add_argument("--assessment", default="", help="Optional path to mams_assessment.json")
    ap.add_argument("--out", default="", help="Optional output Markdown path")
    ap.add_argument("--snippet", action="store_true", help="Render the top-level published snapshot snippet.")
    args = ap.parse_args()

    suite_dir, suite_path = _resolve_suite_path(args)
    assessment_path = Path(args.assessment).resolve() if str(args.assessment).strip() else None
    suite_obj = _load_json(suite_path)
    assessment = _load_json(assessment_path) if assessment_path and assessment_path.exists() else None

    if args.snippet:
        if not str(args.out).strip():
            raise SystemExit("--out is required with --snippet")
        report_text = _render_snapshot_snippet(suite_dir, suite_obj, assessment)
        out_path = Path(args.out).resolve()
    else:
        report_text = _render_detailed_report(suite_dir, suite_obj)
        out_path = Path(args.out).resolve() if str(args.out).strip() else suite_dir / "mams_benchmark_report.md"

    out_path.parent.mkdir(parents=True, exist_ok=True)
    out_path.write_text(report_text + "\n", encoding="utf-8")
    print(report_text)
    return 0


if __name__ == "__main__":
    raise SystemExit(main())
