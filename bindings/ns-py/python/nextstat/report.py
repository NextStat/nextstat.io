"""TREx-like report renderer (publication-ready PDF + per-plot SVG/PNG).

This module is intentionally "view-only": it renders plots from JSON artifacts
emitted by the Rust CLI (`nextstat report` / `nextstat viz ...`).
"""

from __future__ import annotations

import argparse
import hashlib
import json
import math
from dataclasses import dataclass
from pathlib import Path
from typing import Any, Mapping, Sequence


def _require_matplotlib():
    try:
        import matplotlib  # noqa: F401
    except Exception as e:  # pragma: no cover
        raise ImportError("Missing dependency: matplotlib. Install via `pip install nextstat[viz]`.") from e


def _read_json(path: Path) -> dict:
    return json.loads(path.read_text())


def _as_float_list(xs: Sequence[Any]) -> list[float]:
    return [float(x) for x in xs]


def _centers_and_widths(edges: Sequence[float]) -> tuple[list[float], list[float]]:
    e = list(map(float, edges))
    if len(e) < 2:
        return ([], [])
    centers: list[float] = []
    widths: list[float] = []
    for i in range(len(e) - 1):
        centers.append(0.5 * (e[i] + e[i + 1]))
        widths.append(e[i + 1] - e[i])
    return (centers, widths)


@dataclass(frozen=True)
class _SaveTargets:
    pdf: Path
    svg_dir: Path | None
    png_dir: Path | None
    png_dpi: int


def _save_figure(fig, name: str, targets: _SaveTargets, pdf_pages):
    # One page in the PDF.
    pdf_pages.savefig(fig)
    # Optional per-plot SVG.
    if targets.svg_dir is not None:
        targets.svg_dir.mkdir(parents=True, exist_ok=True)
        fig.savefig(targets.svg_dir / f"{name}.svg", format="svg")
    # Optional per-plot PNG.
    if targets.png_dir is not None:
        targets.png_dir.mkdir(parents=True, exist_ok=True)
        fig.savefig(targets.png_dir / f"{name}.png", format="png", dpi=max(72, int(targets.png_dpi)))


_COL = {
    "ink": "#111827",
    "muted": "#4B5563",
    "grid": "#CBD5E1",
    "ratio_band": "#DBEAFE",
    "signal": "#DC2626",
    "data": "#111827",
    "prefit": "#64748B",
    "postfit": "#0F172A",
    "blinded": "#B91C1C",
}

# A deterministic, color-blind-friendly categorical palette inspired by HEP publication figures.
_SAMPLE_PALETTE = [
    "#4C78A8",
    "#F58518",
    "#54A24B",
    "#E45756",
    "#72B7B2",
    "#B279A2",
    "#EECA3B",
    "#FF9DA6",
    "#9D755D",
    "#BAB0AC",
]

_TABLEAU10 = [
    "#4E79A7",
    "#F28E2B",
    "#E15759",
    "#76B7B2",
    "#59A14F",
    "#EDC949",
    "#AF7AA1",
    "#FF9DA7",
    "#9C755F",
    "#BAB0AB",
]

_STYLE: dict[str, Any] = {
    "label_status": "Internal",
    "sqrt_s_tev": 13.0,
    "show_mc_band": True,
    "show_stat_band": True,
    "band_hatch": "////",
    "palette": "hep2026",
}


def _to_bool(v: Any) -> bool:
    if isinstance(v, bool):
        return v
    s = str(v).strip().lower()
    return s in {"1", "true", "yes", "on", "y"}


def _set_render_style(
    *,
    label_status: str | None = None,
    sqrt_s_tev: float | None = None,
    show_mc_band: bool | str | None = None,
    show_stat_band: bool | str | None = None,
    band_hatch: str | None = None,
    palette: str | None = None,
) -> None:
    if label_status is not None:
        _STYLE["label_status"] = str(label_status)
    if sqrt_s_tev is not None:
        _STYLE["sqrt_s_tev"] = float(sqrt_s_tev)
    if show_mc_band is not None:
        _STYLE["show_mc_band"] = _to_bool(show_mc_band)
    if show_stat_band is not None:
        _STYLE["show_stat_band"] = _to_bool(show_stat_band)
    if band_hatch is not None:
        h = str(band_hatch)
        _STYLE["band_hatch"] = h if h else "////"
    if palette is not None:
        p = str(palette).strip().lower()
        _STYLE["palette"] = p if p in {"hep2026", "tableau10"} else "hep2026"


def _sample_color(name: str) -> str:
    n = str(name).strip().lower()
    if "signal" in n or n in {"sig", "s"}:
        return _COL["signal"]
    digest = hashlib.sha1(n.encode("utf-8")).hexdigest()
    palette = _SAMPLE_PALETTE if str(_STYLE.get("palette", "hep2026")) == "hep2026" else _TABLEAU10
    idx = int(digest[:8], 16) % len(palette)
    return palette[idx]


def _add_hep_header(ax, *, status: str = "Internal", subtitle: str | None = None):
    status = str(_STYLE.get("label_status", status))
    sqrt_s = float(_STYLE.get("sqrt_s_tev", 13.0))
    ax.text(
        0.0,
        1.02,
        "NEXTSTAT",
        transform=ax.transAxes,
        ha="left",
        va="bottom",
        fontsize=12,
        fontweight="bold",
        color=_COL["ink"],
    )
    ax.text(
        0.165,
        1.02,
        f"{status}  $\\sqrt{{s}}={sqrt_s:g}$ TeV",
        transform=ax.transAxes,
        ha="left",
        va="bottom",
        fontsize=10,
        color=_COL["muted"],
    )
    if subtitle:
        ax.text(
            1.0,
            1.02,
            subtitle,
            transform=ax.transAxes,
            ha="right",
            va="bottom",
            fontsize=9,
            color=_COL["muted"],
        )


def _plot_total_line(ax, edges: Sequence[float], centers: Sequence[float], y: Sequence[float], **kwargs):
    if len(y) != len(centers):
        return
    if hasattr(ax, "stairs") and len(edges) == len(centers) + 1:
        ax.stairs(y, edges, baseline=None, fill=False, **kwargs)
    else:
        ax.plot(centers, y, **kwargs)


def _extract_band(channel: Mapping[str, Any], key: str, n_bins: int) -> tuple[list[float], list[float]] | None:
    band = channel.get(key)
    if not isinstance(band, Mapping):
        return None
    lo = _as_float_list(band.get("lo") or [])
    hi = _as_float_list(band.get("hi") or [])
    if len(lo) != n_bins or len(hi) != n_bins:
        return None
    return (lo, hi)


def _apply_pub_style():
    import matplotlib as mpl

    # Report rendering is a headless operation; forcing a non-interactive backend
    # avoids crashes in environments without a GUI (e.g. CI, remote shells).
    mpl.use("Agg", force=True)

    mpl.rcParams.update(
        {
            "figure.constrained_layout.use": True,
            "savefig.bbox": "tight",
            "figure.facecolor": "#FFFFFF",
            "axes.facecolor": "#FFFFFF",
            "font.family": ["STIXGeneral", "DejaVu Sans"],
            "mathtext.fontset": "stix",
            "font.size": 10.0,
            "axes.labelsize": 11.0,
            "axes.titlesize": 12.0,
            "axes.linewidth": 1.1,
            "axes.grid": True,
            "grid.color": _COL["grid"],
            "grid.alpha": 0.55,
            "grid.linewidth": 0.6,
            "axes.spines.top": True,
            "axes.spines.right": True,
            "xtick.direction": "in",
            "ytick.direction": "in",
            "xtick.top": True,
            "ytick.right": True,
            "xtick.major.size": 5.0,
            "ytick.major.size": 5.0,
            "xtick.minor.size": 2.5,
            "ytick.minor.size": 2.5,
            "legend.frameon": False,
            "legend.fontsize": 8.5,
            # Keep text as text in SVG, and embed fonts in PDF.
            "svg.fonttype": "none",
            "pdf.fonttype": 42,
            "ps.fonttype": 42,
        }
    )


def _plot_distributions_channel(channel: Mapping[str, Any]):
    import matplotlib.pyplot as plt
    from matplotlib.gridspec import GridSpec

    edges = _as_float_list(channel.get("bin_edges") or [])
    x, w = _centers_and_widths(edges)
    if not x:
        fig, ax = plt.subplots(figsize=(7.6, 3.2))
        ax.text(0.5, 0.5, "Missing binning in distributions artifact", ha="center", va="center")
        ax.axis("off")
        return fig

    blinded = bool(channel.get("data_is_blinded") or False)
    data_y = _as_float_list(channel.get("data_y") or [])
    yerr_lo = _as_float_list(channel.get("data_yerr_lo") or [0.0] * len(data_y))
    yerr_hi = _as_float_list(channel.get("data_yerr_hi") or [0.0] * len(data_y))
    total_prefit = _as_float_list(channel.get("total_prefit_y") or [])
    total_postfit = _as_float_list(channel.get("total_postfit_y") or [])
    ch_name = str(channel.get("channel_name", "channel"))
    mc_band_postfit = _extract_band(channel, "mc_band_postfit", len(x))
    mc_band_postfit_stat = _extract_band(channel, "mc_band_postfit_stat", len(x))
    ratio_band = _extract_band(channel, "ratio_band", len(x))
    ratio_band_stat = _extract_band(channel, "ratio_band_stat", len(x))

    samples = list(channel.get("samples") or [])
    stack_order = channel.get("stack_order")
    if isinstance(stack_order, list) and stack_order:
        order = [str(n) for n in stack_order]
        name_to_series = {str(s.get("name")): s for s in samples}
        samples = [name_to_series[n] for n in order if n in name_to_series]

    fig = plt.figure(figsize=(7.8, 5.6))
    gs = GridSpec(2, 1, height_ratios=[3.3, 1.2], hspace=0.04, figure=fig)
    ax = fig.add_subplot(gs[0])
    axr = fig.add_subplot(gs[1], sharex=ax)
    _add_hep_header(ax, subtitle=f"Channel: {ch_name}")

    bottom = [0.0] * len(x)
    for s in samples:
        name = str(s.get("name", "sample"))
        y = _as_float_list(s.get("postfit_y") or [])
        if len(y) != len(x):
            continue
        ax.bar(
            x,
            y,
            width=w,
            bottom=bottom,
            label=name,
            align="center",
            linewidth=0.35,
            edgecolor="#FFFFFF",
            color=_sample_color(name),
            alpha=0.96,
        )
        bottom = [b + yy for b, yy in zip(bottom, y)]

    _plot_total_line(
        ax,
        edges,
        x,
        total_postfit,
        color=_COL["postfit"],
        linewidth=1.8,
        label="Total postfit",
        zorder=6,
    )
    _plot_total_line(
        ax,
        edges,
        x,
        total_prefit,
        color=_COL["prefit"],
        linewidth=1.2,
        linestyle="--",
        label="Total prefit",
        zorder=5,
    )
    if _to_bool(_STYLE.get("show_mc_band", True)) and mc_band_postfit is not None:
        b_lo, b_hi = mc_band_postfit
        heights = [max(0.0, hi - lo) for lo, hi in zip(b_lo, b_hi)]
        ax.bar(
            x,
            heights,
            bottom=b_lo,
            width=w,
            align="center",
            color="none",
            edgecolor="#334155",
            hatch=str(_STYLE.get("band_hatch", "////")),
            linewidth=0.0,
            alpha=0.55,
            label="MC unc. (total)",
            zorder=7,
        )
    if _to_bool(_STYLE.get("show_stat_band", True)) and mc_band_postfit_stat is not None:
        s_lo, s_hi = mc_band_postfit_stat
        s_h = [max(0.0, hi - lo) for lo, hi in zip(s_lo, s_hi)]
        ax.bar(
            x,
            s_h,
            bottom=s_lo,
            width=[ww * 0.66 for ww in w],
            align="center",
            color="none",
            edgecolor="#0EA5E9",
            hatch="....",
            linewidth=0.0,
            alpha=0.55,
            label="MC unc. (stat)",
            zorder=7,
        )

    if (not blinded) and len(data_y) == len(x):
        ax.errorbar(
            x,
            data_y,
            yerr=[yerr_lo, yerr_hi],
            fmt="o",
            color=_COL["data"],
            markerfacecolor="white",
            markeredgewidth=0.9,
            markersize=3.3,
            linewidth=1.1,
            capsize=0.0,
            label="Data",
            zorder=8,
        )
    if blinded:
        ax.text(
            0.98,
            0.92,
            "BLINDED",
            transform=ax.transAxes,
            ha="right",
            va="top",
            fontsize=10.5,
            color=_COL["blinded"],
            fontweight="bold",
        )

    ymax = 0.0
    for ys in (bottom, data_y, total_postfit, total_prefit):
        for v in ys:
            if isinstance(v, (int, float)) and math.isfinite(float(v)):
                ymax = max(ymax, float(v))
    ax.set_ylim(0.0, max(1.0, 1.35 * ymax))
    ax.set_ylabel("Events / bin")
    ax.set_title(ch_name)
    ax.grid(True, axis="y", alpha=0.55)
    ax.grid(False, axis="x")
    ax.margins(x=0.0)

    handles, labels = ax.get_legend_handles_labels()
    if labels:
        sample_labels = {str(s.get("name", "")).strip().lower() for s in samples}
        items = sorted(zip(handles, labels), key=lambda t: _legend_sort_key(t[1], sample_labels))
        ax.legend(
            [h for (h, _l) in items],
            [l for (_h, l) in items],
            ncol=2,
            loc="upper left",
            columnspacing=1.0,
            handlelength=1.8,
            fontsize=8.2,
        )

    ratio_y = _as_float_list(channel.get("ratio_y") or [])
    ry_lo = _as_float_list(channel.get("ratio_yerr_lo") or [0.0] * len(ratio_y))
    ry_hi = _as_float_list(channel.get("ratio_yerr_hi") or [0.0] * len(ratio_y))
    if (not blinded) and len(ratio_y) == len(x):
        if _to_bool(_STYLE.get("show_mc_band", True)) and ratio_band is not None:
            rb_lo, rb_hi = ratio_band
            rb_h = [max(0.0, hi - lo) for lo, hi in zip(rb_lo, rb_hi)]
            axr.bar(
                x,
                rb_h,
                bottom=rb_lo,
                width=w,
                align="center",
                color="none",
                edgecolor="#1D4ED8",
                hatch=str(_STYLE.get("band_hatch", "////")),
                linewidth=0.0,
                alpha=0.5,
                zorder=1,
            )
        elif ratio_band is None:
            axr.axhspan(0.9, 1.1, color=_COL["ratio_band"], alpha=0.45, zorder=0)
        if _to_bool(_STYLE.get("show_stat_band", True)) and ratio_band_stat is not None:
            rs_lo, rs_hi = ratio_band_stat
            rs_h = [max(0.0, hi - lo) for lo, hi in zip(rs_lo, rs_hi)]
            axr.bar(
                x,
                rs_h,
                bottom=rs_lo,
                width=[ww * 0.66 for ww in w],
                align="center",
                color="none",
                edgecolor="#0EA5E9",
                hatch="....",
                linewidth=0.0,
                alpha=0.5,
                zorder=2,
            )
        axr.axhline(1.0, color=_COL["muted"], lw=1.2, ls="-")
        axr.errorbar(
            x,
            ratio_y,
            yerr=[ry_lo, ry_hi],
            fmt="o",
            color=_COL["data"],
            markerfacecolor="white",
            markeredgewidth=0.9,
            markersize=3.2,
            linewidth=1.1,
            capsize=0.0,
        )

        finite = [float(v) for v in ratio_y if isinstance(v, (int, float)) and math.isfinite(float(v))]
        if finite:
            lo = min(finite + [1.0])
            hi = max(finite + [1.0])
            span = max(0.18, 0.22 * (hi - lo if hi > lo else 1.0))
            ymin = max(0.0, lo - span)
            ymax_r = min(2.4, hi + span)
            if ymax_r - ymin < 0.6:
                ymin = max(0.0, 1.0 - 0.35)
                ymax_r = 1.0 + 0.35
            axr.set_ylim(ymin, ymax_r)
        else:
            axr.set_ylim(0.5, 1.5)

        axr.set_ylabel("Data / MC")
        axr.set_xlabel("Observable bin")
        axr.grid(True, axis="y", alpha=0.4)
        axr.grid(False, axis="x")
        axr.margins(x=0.0)
    else:
        axr.set_visible(False)

    # Hide x tick labels on the top panel.
    plt.setp(ax.get_xticklabels(), visible=False)
    return fig


def _legend_sort_key(label: str, sample_labels: set[str] | None = None) -> tuple[int, str]:
    """Stable legend order close to HEP publication conventions."""
    l = str(label).strip().lower()
    sample_labels = sample_labels or set()
    if l == "data":
        return (0, l)
    if "postfit" in l:
        return (1, l)
    if "prefit" in l:
        return (2, l)
    if "signal" in l:
        return (3, l)
    if l in sample_labels:
        return (4, l)
    if "mc unc." in l and "stat" in l:
        return (6, l)
    if "mc unc." in l and ("total" in l or "tot" in l):
        return (7, l)
    if "mc unc." in l:
        return (8, l)
    return (5, l)


def _plot_pulls(entries: Sequence[Mapping[str, Any]], *, title: str, page: int, n_pages: int):
    import matplotlib.pyplot as plt

    names = [str(e.get("name", "")) for e in entries]
    pulls = [float(e.get("pull", 0.0)) for e in entries]
    constraints = [float(e.get("constraint", 0.0)) for e in entries]

    y = list(range(len(entries)))
    fig_h = max(3.0, 0.18 * len(entries) + 1.4)
    fig, ax = plt.subplots(figsize=(7.6, fig_h))
    _add_hep_header(ax, subtitle=f"Nuisance parameters: {len(entries)}")

    ax.axvspan(-1.0, 1.0, color="#86EFAC", alpha=0.22, zorder=0)
    ax.axvspan(-2.0, 2.0, color="#FDE68A", alpha=0.20, zorder=0)
    ax.axvline(0.0, color=_COL["muted"], lw=1.1)

    ax.errorbar(
        pulls,
        y,
        xerr=constraints,
        fmt="o",
        color=_COL["ink"],
        markerfacecolor="white",
        markeredgewidth=0.9,
        ecolor=_COL["ink"],
        elinewidth=1.2,
        capsize=0.0,
    )

    ax.set_yticks(y)
    ax.set_yticklabels(names, fontsize=8)
    ax.invert_yaxis()
    ax.set_xlabel("Pull (post - pre) / pre_sigma")
    ax.set_xlim(-3.5, 3.5)
    if n_pages > 1:
        ax.set_title(f"{title} (page {page}/{n_pages})")
    else:
        ax.set_title(title)
    ax.grid(True, axis="x", alpha=0.45)
    ax.grid(False, axis="y")
    return fig


def _plot_corr(artifact: Mapping[str, Any]):
    import matplotlib.pyplot as plt

    # Reuse the pure-view plot helper (vector-friendly via pcolormesh).
    from .viz import plot_corr_matrix

    names = artifact.get("parameter_names") or []
    corr = artifact.get("corr") or []
    if not (isinstance(names, list) and isinstance(corr, list) and names):
        fig, ax = plt.subplots(figsize=(7.6, 3.0))
        ax.text(0.5, 0.5, "Missing/empty corr artifact", ha="center", va="center")
        ax.axis("off")
        return fig

    fig, ax = plt.subplots(figsize=(7.6, 6.2))
    plot_corr_matrix(
        artifact,
        ax=ax,
        title="Correlation matrix",
        order="group_base",
        show_colorbar=True,
        cmap="coolwarm",
    )
    _add_hep_header(ax, subtitle=f"Nuisance parameters: {len(names)}")
    return fig


def _plot_yields_channel(channel: Mapping[str, Any]):
    import matplotlib.pyplot as plt

    ch_name = str(channel.get("channel_name", "channel"))
    blinded = bool(channel.get("data_is_blinded") or False)
    data = float(channel.get("data", 0.0))
    total_prefit = float(channel.get("total_prefit", 0.0))
    total_postfit = float(channel.get("total_postfit", 0.0))
    samples = list(channel.get("samples") or [])

    rows: list[list[str]] = []
    for s in samples:
        rows.append(
            [
                str(s.get("name", "")),
                f'{float(s.get("prefit", 0.0)):.3g}',
                f'{float(s.get("postfit", 0.0)):.3g}',
            ]
        )
    rows.append(["Total", f"{total_prefit:.3g}", f"{total_postfit:.3g}"])
    rows.append(["Data", ("blinded" if blinded else f"{data:.3g}"), ""])

    fig, ax = plt.subplots(figsize=(7.6, max(2.8, 0.22 * len(rows) + 1.2)))
    ax.axis("off")
    _add_hep_header(ax, subtitle=f"Channel: {ch_name}")
    ax.set_title(f"Yields: {ch_name}")
    tbl = ax.table(
        cellText=rows,
        colLabels=["Sample", "Prefit", "Postfit"],
        loc="center",
        cellLoc="left",
        colLoc="left",
    )
    tbl.auto_set_font_size(False)
    tbl.set_fontsize(8)
    tbl.scale(1.0, 1.2)
    return fig


def _plot_uncertainty(artifact: Mapping[str, Any]):
    import matplotlib.pyplot as plt

    groups = list(artifact.get("groups") or [])
    if not groups:
        fig, ax = plt.subplots(figsize=(7.6, 2.8))
        ax.text(0.5, 0.5, "Missing/empty uncertainty breakdown", ha="center", va="center")
        ax.axis("off")
        return fig

    names = [str(g.get("name", "")) for g in groups]
    impacts = [float(g.get("impact", 0.0)) for g in groups]
    y = list(range(len(groups)))

    fig_h = max(3.0, 0.22 * len(groups) + 1.2)
    fig, ax = plt.subplots(figsize=(7.6, fig_h))
    _add_hep_header(ax, subtitle=f"Groups: {len(groups)}")
    ax.barh(y, impacts, color="#1D4ED8", alpha=0.88, edgecolor="#1E3A8A", linewidth=0.4)
    ax.set_yticks(y)
    ax.set_yticklabels(names, fontsize=9)
    ax.invert_yaxis()
    ax.set_xlabel("Impact on POI (quadrature)")
    ax.set_title("Uncertainty breakdown")

    try:
        total = float(artifact.get("total", 0.0))
        ax.axvline(total, color="#111827", lw=1.2, ls=":", label=f"Total={total:.3g}")
        ax.legend()
    except Exception:
        pass

    ax.grid(True, axis="x", alpha=0.45)
    ax.grid(False, axis="y")
    return fig


def render_report(
    input_dir: Path | str,
    *,
    pdf: Path | str,
    svg_dir: Path | str | None,
    png_dir: Path | str | None = None,
    png_dpi: int = 220,
    label_status: str = "Internal",
    sqrt_s_tev: float = 13.0,
    show_mc_band: bool = True,
    show_stat_band: bool = True,
    band_hatch: str = "////",
    palette: str = "hep2026",
    corr_include: str | None = None,
    corr_exclude: str | None = None,
    corr_top_n: int | None = None,
) -> None:
    _require_matplotlib()
    _apply_pub_style()
    _set_render_style(
        label_status=label_status,
        sqrt_s_tev=float(sqrt_s_tev),
        show_mc_band=show_mc_band,
        show_stat_band=show_stat_band,
        band_hatch=band_hatch,
        palette=palette,
    )

    from matplotlib.backends.backend_pdf import PdfPages
    import matplotlib.pyplot as plt

    input_dir = Path(input_dir).resolve()
    pdf = Path(pdf).resolve()
    if pdf.parent and not pdf.parent.exists():
        pdf.parent.mkdir(parents=True, exist_ok=True)
    if svg_dir is not None:
        svg_dir = Path(svg_dir).resolve()
    if png_dir is not None:
        png_dir = Path(png_dir).resolve()
    elif svg_dir is not None:
        # Keep backward compatibility for CLI: when only --svg-dir is provided,
        # emit both vector (SVG) and raster (PNG) side-by-side.
        png_dir = svg_dir

    targets = _SaveTargets(pdf=pdf, svg_dir=svg_dir, png_dir=png_dir, png_dpi=int(png_dpi))

    distributions_path = input_dir / "distributions.json"
    pulls_path = input_dir / "pulls.json"
    corr_path = input_dir / "corr.json"
    yields_path = input_dir / "yields.json"
    uncertainty_path = input_dir / "uncertainty.json"

    with PdfPages(pdf) as pages:
        n_saved = 0

        def save(fig, name: str):
            nonlocal n_saved
            _save_figure(fig, name, targets, pages)
            n_saved += 1

        # Distributions: one page per channel.
        if distributions_path.exists():
            d = _read_json(distributions_path)
            for ch in d.get("channels") or []:
                fig = _plot_distributions_channel(ch)
                ch_name = str(ch.get("channel_name", "channel"))
                save(fig, f"distributions__{ch_name}")
                plt.close(fig)

        # Pulls: split into pages if long.
        if pulls_path.exists():
            p = _read_json(pulls_path)
            entries = list(p.get("entries") or [])
            page_size = 60
            if entries:
                n_pages = (len(entries) + page_size - 1) // page_size
                for i in range(n_pages):
                    chunk = entries[i * page_size : (i + 1) * page_size]
                    fig = _plot_pulls(chunk, title="Pulls + constraints", page=i + 1, n_pages=n_pages)
                    save(fig, f"pulls__p{i+1}")
                    plt.close(fig)

        # Correlation matrix.
        if corr_path.exists():
            c = _read_json(corr_path)
            if corr_include or corr_exclude or corr_top_n is not None:
                from .viz import corr_subset

                c_view = dict(c)
                c_view.update(corr_subset(c, include=corr_include, exclude=corr_exclude, top_n=corr_top_n, order="group_base"))
                fig = _plot_corr(c_view)
            else:
                fig = _plot_corr(c)
            save(fig, "corr")
            plt.close(fig)

        # Yields: one page per channel.
        if yields_path.exists():
            y = _read_json(yields_path)
            for ch in y.get("channels") or []:
                fig = _plot_yields_channel(ch)
                ch_name = str(ch.get("channel_name", "channel"))
                save(fig, f"yields__{ch_name}")
                plt.close(fig)

        # Uncertainty breakdown (groups).
        if uncertainty_path.exists():
            u = _read_json(uncertainty_path)
            fig = _plot_uncertainty(u)
            save(fig, "uncertainty")
            plt.close(fig)

        # Always emit a valid PDF even if no artifacts were present.
        if n_saved == 0:
            fig, ax = plt.subplots(figsize=(7.6, 2.8), constrained_layout=False)
            ax.text(0.5, 0.5, f"No report artifacts found in: {input_dir}", ha="center", va="center")
            ax.axis("off")
            save(fig, "empty_report")
            plt.close(fig)


def _cmd_render(args: argparse.Namespace) -> int:
    render_report(
        Path(args.input_dir),
        pdf=Path(args.pdf),
        svg_dir=Path(args.svg_dir) if args.svg_dir else None,
        png_dir=Path(args.png_dir) if args.png_dir else None,
        png_dpi=int(args.png_dpi),
        label_status=str(args.label_status),
        sqrt_s_tev=float(args.sqrt_s_tev),
        show_mc_band=_to_bool(args.show_mc_band),
        show_stat_band=_to_bool(args.show_stat_band),
        band_hatch=str(args.band_hatch),
        palette=str(args.palette),
        corr_include=args.corr_include,
        corr_exclude=args.corr_exclude,
        corr_top_n=int(args.corr_top_n) if args.corr_top_n is not None else None,
    )
    return 0


def main(argv: Sequence[str] | None = None) -> int:
    p = argparse.ArgumentParser(prog="python -m nextstat.report")
    sub = p.add_subparsers(dest="cmd", required=True)

    r = sub.add_parser("render", help="Render report PDF (+ optional per-plot SVG/PNG) from artifacts directory")
    r.add_argument("--input-dir", required=True, help="Directory containing artifacts JSON files")
    r.add_argument("--pdf", required=True, help="Output PDF path")
    r.add_argument("--svg-dir", default=None, help="Optional directory for per-plot SVGs")
    r.add_argument(
        "--png-dir",
        default=None,
        help="Optional directory for per-plot PNGs (defaults to --svg-dir when omitted)",
    )
    r.add_argument("--png-dpi", type=int, default=220, help="PNG DPI for per-plot exports")
    r.add_argument("--label-status", default="Internal", help="Header status label (Internal/Preliminary/Public)")
    r.add_argument("--sqrt-s-tev", type=float, default=13.0, help="Center-of-mass energy shown in header")
    r.add_argument("--show-mc-band", default="1", help="Show total MC uncertainty band (1/0)")
    r.add_argument("--show-stat-band", default="1", help="Show stat-only MC uncertainty band (1/0)")
    r.add_argument("--band-hatch", default="////", help="Hatch pattern for total uncertainty band")
    r.add_argument("--palette", default="hep2026", choices=["hep2026", "tableau10"], help="Sample palette")
    r.add_argument("--corr-include", default=None, help="Regex: include parameters for corr plot")
    r.add_argument("--corr-exclude", default=None, help="Regex: exclude parameters for corr plot")
    r.add_argument("--corr-top-n", default=None, help="Keep top-N parameters by max |corr| (after filters)")
    r.set_defaults(fn=_cmd_render)

    args = p.parse_args(list(argv) if argv is not None else None)
    return int(args.fn(args))


if __name__ == "__main__":  # pragma: no cover
    raise SystemExit(main())
