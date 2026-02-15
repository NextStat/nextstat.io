"""Validation report renderer (publishable PDF).

This module renders a human-readable PDF from `validation_report.json`
produced by `nextstat validation-report`.

Design goal: an audit-friendly document that is dependency-light
(matplotlib only, via `nextstat[viz]`).
"""

from __future__ import annotations

import argparse
import datetime
import json
import textwrap
from dataclasses import dataclass
from pathlib import Path
from typing import Any, Mapping, Sequence


def _require_matplotlib():
    try:
        import matplotlib  # noqa: F401
    except Exception as e:  # pragma: no cover
        raise ImportError(
            "Missing dependency: matplotlib. Install via `pip install nextstat[viz]`."
        ) from e


def _apply_pub_style():
    import matplotlib as mpl

    mpl.use("Agg", force=True)
    mpl.rcParams.update(
        {
            "figure.constrained_layout.use": True,
            "savefig.bbox": "tight",
            "axes.grid": False,
            "axes.spines.top": False,
            "axes.spines.right": False,
            "axes.spines.left": False,
            "axes.spines.bottom": False,
            "font.size": 10.0,
            # Keep text as text in SVG, and embed fonts in PDF.
            "svg.fonttype": "none",
            "pdf.fonttype": 42,
            "ps.fonttype": 42,
        }
    )


def _read_json(path: Path) -> dict[str, Any]:
    return json.loads(path.read_text())


def _sha_short(s: str | None, n: int = 12) -> str:
    if not s:
        return "unknown"
    s = str(s)
    return s[:n]


@dataclass(frozen=True)
class _Targets:
    pdf: Path


def _new_page(figsize: tuple[float, float] = (8.27, 11.69)):
    import matplotlib.pyplot as plt

    fig = plt.figure(figsize=figsize)
    ax = fig.add_subplot(111)
    ax.axis("off")
    return fig, ax


_COL = {
    "ink": "#111827",
    "muted": "#6B7280",
    "muted2": "#374151",
    "border": "#E5E7EB",
    "accent": "#1D4ED8",
    "ok": "#16A34A",
    "fail": "#DC2626",
    "skipped": "#6B7280",
    "warn": "#D97706",
    "bg": "#FFFFFF",
    "panel": "#F3F4F6",
}


def _status_color(status: str) -> str:
    s = (status or "").lower()
    if s == "ok":
        return _COL["ok"]
    if s in ("fail", "error"):
        return _COL["fail"]
    if s == "skipped":
        return _COL["skipped"]
    return _COL["muted"]


def _add_title(ax, title: str, subtitle: str | None = None):
    ax.text(
        0.0,
        0.97,
        title,
        ha="left",
        va="top",
        fontsize=18,
        weight="bold",
        color=_COL["ink"],
        transform=ax.transAxes,
    )
    if subtitle:
        ax.text(
            0.0,
            0.93,
            subtitle,
            ha="left",
            va="top",
            fontsize=11,
            color=_COL["muted2"],
            transform=ax.transAxes,
        )


def _add_kv_block(ax, y0: float, kv: list[tuple[str, str]]):
    y = y0
    for k, v in kv:
        ax.text(
            0.0,
            y,
            f"{k}:",
            ha="left",
            va="top",
            family="monospace",
            color=_COL["muted2"],
            transform=ax.transAxes,
        )
        ax.text(
            0.28,
            y,
            v,
            ha="left",
            va="top",
            family="monospace",
            color=_COL["ink"],
            transform=ax.transAxes,
        )
        y -= 0.032


def _add_footer(ax, page: int, total: int, *, ws_sha: str, apex_sha: str):
    ax.text(
        0.0,
        0.02,
        f"ws={ws_sha}  apex2={apex_sha}",
        ha="left",
        va="bottom",
        fontsize=8,
        family="monospace",
        color=_COL["muted"],
        transform=ax.transAxes,
    )
    ax.text(
        1.0,
        0.02,
        f"Page {page}/{total}",
        ha="right",
        va="bottom",
        fontsize=8,
        color=_COL["muted"],
        transform=ax.transAxes,
    )


def _pill(ax, x: float, y: float, text: str, color: str):
    # Draw a status "pill" using a rounded bbox around text.
    ax.text(
        x,
        y,
        text,
        ha="left",
        va="center",
        fontsize=11,
        weight="bold",
        color="#FFFFFF",
        transform=ax.transAxes,
        bbox={
            "boxstyle": "round,pad=0.35,rounding_size=0.15",
            "facecolor": color,
            "edgecolor": color,
        },
    )


def _card(ax, x: float, y: float, w: float, h: float, *, title: str, body: str):
    """Document-like panel with a title and wrapped body text."""
    from matplotlib.patches import FancyBboxPatch

    ax.add_patch(
        FancyBboxPatch(
            (x, y),
            w,
            h,
            boxstyle="round,pad=0.012,rounding_size=0.02",
            transform=ax.transAxes,
            facecolor=_COL["panel"],
            edgecolor=_COL["border"],
            linewidth=1.0,
            zorder=0,
        )
    )
    ax.text(
        x + 0.02,
        y + h - 0.05,
        title,
        ha="left",
        va="top",
        fontsize=10,
        weight="bold",
        color=_COL["muted2"],
        transform=ax.transAxes,
    )
    ax.text(
        x + 0.02,
        y + h - 0.09,
        _wrap(body, width=82),
        ha="left",
        va="top",
        fontsize=9.5,
        color=_COL["ink"],
        transform=ax.transAxes,
        linespacing=1.25,
    )


def _get_suites(report: Mapping[str, Any]) -> dict[str, Any]:
    apex = report.get("apex2_summary") or {}
    suites = apex.get("suites") or {}
    return suites if isinstance(suites, dict) else {}


def _get_suite(report: Mapping[str, Any], name: str) -> dict[str, Any]:
    s = _get_suites(report).get(name) or {}
    return s if isinstance(s, dict) else {}


def _get_worst_cases(suite: Mapping[str, Any], metric: str) -> list[dict[str, Any]]:
    xs = suite.get("worst_cases") or []
    if not isinstance(xs, list):
        return []
    out: list[dict[str, Any]] = []
    for it in xs:
        if not isinstance(it, dict):
            continue
        if str(it.get("metric") or "") != metric:
            continue
        out.append(it)
    out.sort(key=lambda d: float(d.get("value") or 0.0), reverse=True)
    return out


def _suite_status_counts(suites: Mapping[str, Any]) -> dict[str, int]:
    counts = {"ok": 0, "fail": 0, "skipped": 0, "unknown": 0}
    for _k, v in suites.items():
        if not isinstance(v, Mapping):
            counts["unknown"] += 1
            continue
        st = str(v.get("status") or "unknown").lower()
        if st in ("ok", "fail", "skipped"):
            counts[st] += 1
        else:
            counts["unknown"] += 1
    return counts


def _pick_suite(report: Mapping[str, Any], names: Sequence[str]) -> tuple[str, dict[str, Any]]:
    suites = _get_suites(report)
    for n in names:
        v = suites.get(n)
        if isinstance(v, dict):
            return str(n), v
    return str(names[0] if names else "unknown"), {}


def _fmt_sci(x: Any) -> str:
    try:
        v = float(x)
    except Exception:
        return "-"
    if not (v == v):  # NaN
        return "NaN"
    return f"{v:.3e}"


def _wrap(s: Any, width: int = 92) -> str:
    s = str(s or "")
    return "\n".join(textwrap.wrap(s, width=width)) if s else ""


def _render_pdf(report: Mapping[str, Any], targets: _Targets):
    _require_matplotlib()
    _apply_pub_style()

    from matplotlib.backends.backend_pdf import PdfPages
    from matplotlib.patches import Rectangle

    schema_version = str(report.get("schema_version") or "")
    overall = str(((report.get("apex2_summary") or {}).get("overall")) or "unknown")
    det = bool(report.get("deterministic") is True)

    ds = report.get("dataset_fingerprint") or {}
    env = report.get("environment") or {}
    apex = report.get("apex2_summary") or {}
    suites = _get_suites(report)
    rr = report.get("regulated_review") or {}

    ws_sha = _sha_short(ds.get("workspace_sha256"))
    apex_sha = _sha_short(apex.get("master_report_sha256"))

    targets.pdf.parent.mkdir(parents=True, exist_ok=True)
    # Matplotlib writes timestamps into PDF metadata by default. In deterministic mode,
    # pin CreationDate/ModDate so the output is stable across runs.
    metadata: dict[str, Any] | None = None
    if det:
        fixed_dt = datetime.datetime(1970, 1, 1, 0, 0, 0, tzinfo=datetime.timezone.utc)
        metadata = {
            "Title": "NextStat Validation Report",
            "Creator": "nextstat.validation_report",
            "Producer": "NextStat (matplotlib)",
            "CreationDate": fixed_dt,
            "ModDate": fixed_dt,
        }

    # Fixed report structure. If you add/remove pages, update this number.
    total_pages = 8
    page = 1

    with PdfPages(targets.pdf, metadata=metadata) as pages:
        # Page 1: cover
        fig, ax = _new_page()
        # Subtle top band to read more like a document cover.
        ax.add_patch(
            Rectangle(
                (0.0, 0.91),
                1.0,
                0.09,
                transform=ax.transAxes,
                facecolor=_COL["panel"],
                edgecolor="none",
                zorder=0,
            )
        )
        _add_title(
            ax,
            "NextStat Validation Report",
            subtitle="Validation pack: suite outcomes + deterministic fingerprints",
        )
        status_color = _status_color(overall)
        _pill(ax, 0.0, 0.86, overall.upper(), status_color)

        counts = _suite_status_counts(suites)

        key_lines: list[str] = []
        for suite_name in ("pyhf", "nuts_quality", "nuts_quality_report", "root"):
            if suite_name not in suites:
                continue
            s = suites.get(suite_name) or {}
            st = str((s or {}).get("status") or "unknown")
            key_lines.append(f"{suite_name}: {st}")
        key_block = "\n".join(key_lines) if key_lines else "No key suites recorded."

        contains_raw = (
            str(rr.get("contains_raw_data")).lower()
            if isinstance(rr, Mapping) and "contains_raw_data" in rr
            else "unknown"
        )
        _card(
            ax,
            0.0,
            0.58,
            0.64,
            0.24,
            title="Executive Summary",
            body=(
                f"Suites: ok={counts['ok']}  fail={counts['fail']}  skipped={counts['skipped']}  unknown={counts['unknown']}\n"
                f"Deterministic: {det}    Contains raw data: {contains_raw}\n\n"
                f"Key suites:\n{key_block}"
            ),
        )
        _card(
            ax,
            0.68,
            0.58,
            0.32,
            0.24,
            title="Artifacts",
            body=(
                f"Workspace SHA-256: {ws_sha}\n"
                f"Apex2 master SHA-256: {apex_sha}\n\n"
                f"NextStat: {str(env.get('nextstat_version') or 'unknown')}\n"
                f"Platform: {str(env.get('platform') or 'unknown')}"
            ),
        )
        _card(
            ax,
            0.0,
            0.30,
            1.0,
            0.22,
            title="What This Report Is",
            body=(
                "A compact, audit-friendly summary of Apex2 validation outcomes tied to a specific workspace fingerprint. "
                "It reports suite-level pass/fail and worst-case deltas for key suites."
            ),
        )
        ax.text(
            0.0,
            0.08,
            f"Generated from validation_report.json (schema={schema_version}).",
            ha="left",
            va="bottom",
            fontsize=9,
            color=_COL["muted"],
            transform=ax.transAxes,
        )
        _add_footer(ax, page, total_pages, ws_sha=ws_sha, apex_sha=apex_sha)
        pages.savefig(fig)
        page += 1

        # Page 2: suite matrix
        fig, ax = _new_page()
        _add_title(ax, "Validation Matrix", subtitle="Suite-level pass/fail status (Apex2)")
        counts = _suite_status_counts(suites)
        ax.text(
            0.0,
            0.89,
            f"ok={counts['ok']}  fail={counts['fail']}  skipped={counts['skipped']}  unknown={counts['unknown']}",
            ha="left",
            va="top",
            fontsize=10,
            color=_COL["muted2"],
            transform=ax.transAxes,
        )

        suite_names = sorted(map(str, suites.keys()))
        cell_text: list[list[str]] = []
        cell_colors: list[list[str]] = []
        for n in suite_names:
            s = suites.get(n) or {}
            st = str(s.get("status") or "unknown")
            cell_text.append([n, st])
            cell_colors.append([_COL["bg"], _status_color(st)])

        if suite_names:
            tbl = ax.table(
                cellText=cell_text,
                cellColours=cell_colors,
                colLabels=["Suite", "Status"],
                colColours=[_COL["border"], _COL["border"]],
                loc="upper left",
                bbox=[0.0, 0.16, 1.0, 0.72],
            )
            tbl.auto_set_font_size(False)
            tbl.set_fontsize(10)
            # Improve readability: left-align suite names, white text on colored status.
            for (r, c), cell in tbl.get_celld().items():
                cell.set_edgecolor(_COL["border"])
                if r == 0:
                    cell.get_text().set_weight("bold")
                    cell.get_text().set_color(_COL["ink"])
                else:
                    if c == 0:
                        cell.get_text().set_ha("left")
                        cell.get_text().set_color(_COL["ink"])
                    if c == 1:
                        cell.get_text().set_color("#FFFFFF")
                        cell.get_text().set_weight("bold")
        else:
            ax.text(0.0, 0.85, "No suite data available.", transform=ax.transAxes)

        _add_footer(ax, page, total_pages, ws_sha=ws_sha, apex_sha=apex_sha)
        pages.savefig(fig)
        page += 1

        # Page 3: pyhf worst-case tables
        fig, ax = _new_page()
        _add_title(ax, "Key Suite: pyhf Parity", subtitle="Worst-case deltas (Apex2 pyhf suite)")
        pyhf = _get_suite(report, "pyhf")
        ax.text(
            0.0,
            0.88,
            f"Status: {pyhf.get('status','unknown')}    Cases: {pyhf.get('n_cases','-')}",
            ha="left",
            va="top",
            fontsize=10,
            color=_COL["muted2"],
            transform=ax.transAxes,
        )
        ph = pyhf.get("highlights") if isinstance(pyhf.get("highlights"), list) else []
        if ph:
            ax.text(
                0.0,
                0.84,
                "\n".join(f"- {str(x)}" for x in ph[:4]),
                ha="left",
                va="top",
                fontsize=9,
                color=_COL["ink"],
                transform=ax.transAxes,
            )

        nll_rows = _get_worst_cases(pyhf, "max_abs_delta_nll")[:5]
        exp_rows = _get_worst_cases(pyhf, "max_abs_delta_expected_full")[:5]

        def _rows_to_table(rows: Sequence[Mapping[str, Any]], *, title: str, y_top: float):
            ax.text(0.0, y_top, title, ha="left", va="top", fontsize=11, weight="bold", transform=ax.transAxes)
            data = []
            for r in rows:
                data.append(
                    [
                        str(r.get("case") or "unknown"),
                        _fmt_sci(r.get("value")),
                        str(r.get("notes") or ""),
                    ]
                )
            if not data:
                data = [["-", "-", ""]]
            tbl = ax.table(
                cellText=data,
                colLabels=["Case", "Value", "Notes"],
                colColours=[_COL["border"], _COL["border"], _COL["border"]],
                loc="upper left",
                bbox=[0.0, y_top - 0.34, 1.0, 0.28],
            )
            tbl.auto_set_font_size(False)
            tbl.set_fontsize(9)
            for (r, c), cell in tbl.get_celld().items():
                cell.set_edgecolor(_COL["border"])
                if r == 0:
                    cell.get_text().set_weight("bold")
                if r > 0 and c == 0:
                    cell.get_text().set_ha("left")

        _rows_to_table(nll_rows, title="Top worst |dNLL|", y_top=0.80)
        _rows_to_table(exp_rows, title="Top worst |d expected(full)|", y_top=0.44)

        _add_footer(ax, page, total_pages, ws_sha=ws_sha, apex_sha=apex_sha)
        pages.savefig(fig)
        page += 1

        # Page 4: NUTS diagnostics
        fig, ax = _new_page()
        _add_title(ax, "Key Suite: NUTS Diagnostics", subtitle="Worst-case sampler quality signals (Apex2)")

        nuts_name, nuts = _pick_suite(report, ["nuts_quality", "nuts_quality_report"])
        ax.text(
            0.0,
            0.88,
            f"NUTS diagnostics ({nuts_name})",
            ha="left",
            va="top",
            weight="bold",
            transform=ax.transAxes,
        )
        ax.text(
            0.0,
            0.84,
            f"Status: {nuts.get('status','not present')}",
            ha="left",
            va="top",
            fontsize=10,
            color=_COL["muted2"],
            transform=ax.transAxes,
        )
        nh = nuts.get("highlights") if isinstance(nuts.get("highlights"), list) else []
        if nh:
            ax.text(
                0.0,
                0.79,
                "\n".join(f"- {str(x)}" for x in nh[:5]),
                ha="left",
                va="top",
                fontsize=9,
                color=_COL["ink"],
                transform=ax.transAxes,
            )

        def _metric_row(suite: Mapping[str, Any], metric: str, label: str) -> list[str]:
            rows = _get_worst_cases(suite, metric)
            if not rows:
                return [label, "-", "-", ""]
            r = rows[0]
            return [
                label,
                _fmt_sci(r.get("value")),
                str(r.get("case") or "-"),
                str(r.get("notes") or ""),
            ]

        nuts_rows = [
            _metric_row(nuts, "divergence_rate", "divergence_rate"),
            _metric_row(nuts, "max_treedepth_rate", "max_treedepth_rate"),
            _metric_row(nuts, "max_r_hat", "max_r_hat"),
            _metric_row(nuts, "min_ess_bulk", "min_ess_bulk"),
            _metric_row(nuts, "min_ess_tail", "min_ess_tail"),
            _metric_row(nuts, "min_ebfmi", "min_ebfmi"),
        ]
        tbl = ax.table(
            cellText=nuts_rows,
            colLabels=["Metric", "Value", "Case", "Notes"],
            colColours=[_COL["border"], _COL["border"], _COL["border"], _COL["border"]],
            loc="upper left",
            bbox=[0.0, 0.45, 1.0, 0.28],
        )
        tbl.auto_set_font_size(False)
        tbl.set_fontsize(9)
        for (r, c), cell in tbl.get_celld().items():
            cell.set_edgecolor(_COL["border"])
            if r == 0:
                cell.get_text().set_weight("bold")
            if r > 0 and c == 0:
                cell.get_text().set_ha("left")

        _add_footer(ax, page, total_pages, ws_sha=ws_sha, apex_sha=apex_sha)
        pages.savefig(fig)
        page += 1

        # Page 5: ROOT parity
        fig, ax = _new_page()
        _add_title(ax, "Key Suite: ROOT Parity", subtitle="Worst-case parity deltas (Apex2 root suite)")
        root = _get_suite(report, "root")
        ax.text(
            0.0,
            0.88,
            f"Status: {root.get('status','not present')}    Cases: {root.get('n_cases','-')}    OK: {root.get('n_ok','-')}",
            ha="left",
            va="top",
            fontsize=10,
            color=_COL["muted2"],
            transform=ax.transAxes,
        )

        dq_rows = _get_worst_cases(root, "max_abs_dq_mu")[:5]
        mu_rows = _get_worst_cases(root, "abs_d_mu_hat")[:5]

        def _rows_to_table(rows: Sequence[Mapping[str, Any]], *, title: str, y_top: float):
            ax.text(0.0, y_top, title, ha="left", va="top", fontsize=11, weight="bold", transform=ax.transAxes)
            data = []
            for r in rows:
                data.append(
                    [
                        str(r.get("case") or "unknown"),
                        _fmt_sci(r.get("value")),
                        str(r.get("notes") or ""),
                    ]
                )
            if not data:
                data = [["(not available)", "-", ""]]
            tbl = ax.table(
                cellText=data,
                colLabels=["Case", "Value", "Notes"],
                colColours=[_COL["border"], _COL["border"], _COL["border"]],
                loc="upper left",
                bbox=[0.0, y_top - 0.34, 1.0, 0.28],
            )
            tbl.auto_set_font_size(False)
            tbl.set_fontsize(9)
            for (r, c), cell in tbl.get_celld().items():
                cell.set_edgecolor(_COL["border"])
                if r == 0:
                    cell.get_text().set_weight("bold")
                if r > 0 and c == 0:
                    cell.get_text().set_ha("left")

        _rows_to_table(dq_rows, title="Top worst max_abs_dq_mu", y_top=0.80)
        _rows_to_table(mu_rows, title="Top worst abs d_mu_hat", y_top=0.44)

        _add_footer(ax, page, total_pages, ws_sha=ws_sha, apex_sha=apex_sha)
        pages.savefig(fig)
        page += 1

        # Page 6: Pharma PK/NLME validation
        fig, ax = _new_page()
        _add_title(ax, "Key Suite: Pharma PK/NLME", subtitle="Parameter recovery and population estimation (Apex2)")

        pharma_smoke = _get_suite(report, "pharma")
        pharma_ref_name, pharma_ref = _pick_suite(report, ["pharma_reference"])

        ax.text(
            0.0,
            0.88,
            f"Pharma smoke: {pharma_smoke.get('status', 'not present')}    "
            f"Pharma reference ({pharma_ref_name}): {pharma_ref.get('status', 'not present')}",
            ha="left",
            va="top",
            fontsize=10,
            color=_COL["muted2"],
            transform=ax.transAxes,
        )

        pref_n = pharma_ref.get("n_cases", "-")
        pref_ok = pharma_ref.get("n_ok", "-")
        ax.text(
            0.0,
            0.84,
            f"Reference cases: {pref_ok}/{pref_n} ok",
            ha="left",
            va="top",
            fontsize=10,
            color=_COL["ink"],
            transform=ax.transAxes,
        )

        ph_hl = pharma_ref.get("highlights") if isinstance(pharma_ref.get("highlights"), list) else []
        if ph_hl:
            ax.text(
                0.0,
                0.80,
                "\n".join(f"- {str(x)}" for x in ph_hl[:5]),
                ha="left",
                va="top",
                fontsize=9,
                color=_COL["ink"],
                transform=ax.transAxes,
            )

        param_rows = _get_worst_cases(pharma_ref, "max_abs_param_err")[:5]
        pred_rows = _get_worst_cases(pharma_ref, "max_abs_err")[:5]

        def _pharma_table(rows: Sequence[Mapping[str, Any]], *, title: str, y_top: float):
            ax.text(0.0, y_top, title, ha="left", va="top", fontsize=11, weight="bold", transform=ax.transAxes)
            data = []
            for r in rows:
                data.append(
                    [
                        str(r.get("case") or "unknown"),
                        _fmt_sci(r.get("value")),
                        str(r.get("notes") or ""),
                    ]
                )
            if not data:
                data = [["(not available)", "-", ""]]
            tbl = ax.table(
                cellText=data,
                colLabels=["Case", "Value", "Notes"],
                colColours=[_COL["border"], _COL["border"], _COL["border"]],
                loc="upper left",
                bbox=[0.0, y_top - 0.30, 1.0, 0.24],
            )
            tbl.auto_set_font_size(False)
            tbl.set_fontsize(9)
            for (r, c), cell in tbl.get_celld().items():
                cell.set_edgecolor(_COL["border"])
                if r == 0:
                    cell.get_text().set_weight("bold")
                if r > 0 and c == 0:
                    cell.get_text().set_ha("left")

        _pharma_table(param_rows, title="Worst parameter recovery error", y_top=0.72)
        _pharma_table(pred_rows, title="Worst prediction error (analytic)", y_top=0.38)

        _add_footer(ax, page, total_pages, ws_sha=ws_sha, apex_sha=apex_sha)
        pages.savefig(fig)
        page += 1

        # Page 7: dataset fingerprint (full hashes and counts)  [was page 6]
        fig, ax = _new_page()
        _add_title(ax, "Dataset Fingerprint", subtitle="Workspace hash and structure summary")
        channels = ds.get("channels") or []
        n_bins = ds.get("n_bins_per_channel") or []
        obs = ds.get("observation_summary") or {}
        _add_kv_block(
            ax,
            0.88,
            [
                ("workspace_sha256", str(ds.get("workspace_sha256") or "unknown")),
                ("workspace_bytes", str(ds.get("workspace_bytes") or "unknown")),
                ("n_channels", str(ds.get("n_channels") or len(channels))),
                ("channels", ", ".join(map(str, channels)) if channels else "unknown"),
                ("n_bins_per_channel", ", ".join(map(str, n_bins)) if n_bins else "unknown"),
                ("n_samples_total", str(ds.get("n_samples_total") or "unknown")),
                ("n_parameters", str(ds.get("n_parameters") or "unknown")),
                ("obs_total", str(obs.get("total_observed") or "unknown")),
                ("obs_min_bin", str(obs.get("min_bin") or "unknown")),
                ("obs_max_bin", str(obs.get("max_bin") or "unknown")),
            ],
        )
        _add_footer(ax, page, total_pages, ws_sha=ws_sha, apex_sha=apex_sha)
        pages.savefig(fig)
        page += 1

        # Page 8: environment  [was page 7]
        fig, ax = _new_page()
        _add_title(ax, "Environment", subtitle="Reproducibility metadata")
        det_settings = env.get("determinism_settings") or {}
        _add_kv_block(
            ax,
            0.88,
            [
                ("nextstat_version", str(env.get("nextstat_version") or "unknown")),
                ("nextstat_git_commit", str(env.get("nextstat_git_commit") or "unknown")),
                ("rust_toolchain", str(env.get("rust_toolchain") or "unknown")),
                ("python_version", str(env.get("python_version") or "unknown")),
                ("pyhf_version", str(env.get("pyhf_version") or "unknown")),
                ("platform", str(env.get("platform") or "unknown")),
                ("determinism_settings", json.dumps(det_settings, sort_keys=True)),
            ],
        )
        _add_footer(ax, page, total_pages, ws_sha=ws_sha, apex_sha=apex_sha)
        pages.savefig(fig)

def main(argv: list[str] | None = None) -> int:
    ap = argparse.ArgumentParser(prog="python -m nextstat.validation_report")
    sub = ap.add_subparsers(dest="cmd", required=True)

    ap_render = sub.add_parser("render", help="Render validation_report.json to a PDF.")
    ap_render.add_argument("--input", type=Path, required=True, help="Path to validation_report.json")
    ap_render.add_argument("--pdf", type=Path, required=True, help="Output PDF path")

    args = ap.parse_args(argv)

    if args.cmd == "render":
        report = _read_json(args.input)
        _render_pdf(report, _Targets(pdf=args.pdf))
        return 0

    raise SystemExit("unreachable")  # pragma: no cover


if __name__ == "__main__":  # pragma: no cover
    raise SystemExit(main())
