"""TREx-like report renderer (publication-ready PDF + per-plot SVG).

This module is intentionally "view-only": it renders plots from JSON artifacts
emitted by the Rust CLI (`nextstat report` / `nextstat viz ...`).
"""

from __future__ import annotations

import argparse
import json
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


def _save_figure(fig, name: str, targets: _SaveTargets, pdf_pages):
    # One page in the PDF.
    pdf_pages.savefig(fig)
    # Optional per-plot SVG.
    if targets.svg_dir is not None:
        targets.svg_dir.mkdir(parents=True, exist_ok=True)
        fig.savefig(targets.svg_dir / f"{name}.svg", format="svg")


def _apply_pub_style():
    import matplotlib as mpl

    # Report rendering is a headless operation; forcing a non-interactive backend
    # avoids crashes in environments without a GUI (e.g. CI, remote shells).
    mpl.use("Agg", force=True)

    mpl.rcParams.update(
        {
            "figure.constrained_layout.use": True,
            "savefig.bbox": "tight",
            "axes.grid": True,
            "grid.alpha": 0.25,
            "axes.spines.top": False,
            "axes.spines.right": False,
            "legend.frameon": False,
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
    blinded = bool(channel.get("data_is_blinded") or False)
    data_y = _as_float_list(channel.get("data_y") or [])
    yerr_lo = _as_float_list(channel.get("data_yerr_lo") or [0.0] * len(data_y))
    yerr_hi = _as_float_list(channel.get("data_yerr_hi") or [0.0] * len(data_y))

    samples = list(channel.get("samples") or [])
    stack_order = channel.get("stack_order")
    if isinstance(stack_order, list) and stack_order:
        order = [str(n) for n in stack_order]
        name_to_series = {str(s.get("name")): s for s in samples}
        samples = [name_to_series[n] for n in order if n in name_to_series]

    fig = plt.figure(figsize=(7.6, 5.2))
    gs = GridSpec(2, 1, height_ratios=[3.0, 1.2], hspace=0.05, figure=fig)
    ax = fig.add_subplot(gs[0])
    axr = fig.add_subplot(gs[1], sharex=ax)

    bottom = [0.0] * len(x)
    for s in samples:
        name = str(s.get("name", "sample"))
        y = _as_float_list(s.get("postfit_y") or [])
        if len(y) != len(x):
            continue
        ax.bar(x, y, width=w, bottom=bottom, label=name, align="center", linewidth=0.0, alpha=0.9)
        bottom = [b + yy for b, yy in zip(bottom, y)]

    if (not blinded) and len(data_y) == len(x):
        ax.errorbar(
            x,
            data_y,
            yerr=[yerr_lo, yerr_hi],
            fmt="o",
            color="#111827",
            markersize=3.0,
            linewidth=1.0,
            capsize=0.0,
            label="Data",
            zorder=5,
        )
    if blinded:
        ax.text(
            0.98,
            0.92,
            "BLINDED",
            transform=ax.transAxes,
            ha="right",
            va="top",
            fontsize=10,
            color="#B91C1C",
        )

    ax.set_ylabel("Events / bin")
    ax.set_title(str(channel.get("channel_name", "channel")))
    ax.legend(ncol=2, fontsize=8)

    ratio_y = _as_float_list(channel.get("ratio_y") or [])
    ry_lo = _as_float_list(channel.get("ratio_yerr_lo") or [0.0] * len(ratio_y))
    ry_hi = _as_float_list(channel.get("ratio_yerr_hi") or [0.0] * len(ratio_y))
    if (not blinded) and len(ratio_y) == len(x):
        axr.axhline(1.0, color="#6B7280", lw=1.0, ls=":")
        axr.errorbar(
            x,
            ratio_y,
            yerr=[ry_lo, ry_hi],
            fmt="o",
            color="#111827",
            markersize=3.0,
            linewidth=1.0,
            capsize=0.0,
        )
        axr.set_ylim(0.0, 2.0)
        axr.set_ylabel("Data/MC")
        axr.set_xlabel("Observable")
    else:
        axr.set_visible(False)

    # Hide x tick labels on the top panel.
    plt.setp(ax.get_xticklabels(), visible=False)
    return fig


def _plot_pulls(entries: Sequence[Mapping[str, Any]], *, title: str, page: int, n_pages: int):
    import matplotlib.pyplot as plt

    names = [str(e.get("name", "")) for e in entries]
    pulls = [float(e.get("pull", 0.0)) for e in entries]
    constraints = [float(e.get("constraint", 0.0)) for e in entries]

    y = list(range(len(entries)))
    fig_h = max(3.0, 0.18 * len(entries) + 1.4)
    fig, ax = plt.subplots(figsize=(7.6, fig_h))

    ax.axvspan(-1.0, 1.0, color="#7BD389", alpha=0.20, zorder=0)
    ax.axvspan(-2.0, 2.0, color="#F2D95C", alpha=0.15, zorder=0)
    ax.axvline(0.0, color="#6B7280", lw=1.0)

    ax.errorbar(pulls, y, xerr=constraints, fmt="o", color="#111827", ecolor="#111827", elinewidth=1.2, capsize=0.0)

    ax.set_yticks(y)
    ax.set_yticklabels(names, fontsize=8)
    ax.invert_yaxis()
    ax.set_xlabel("Pull (post - pre) / pre_sigma")
    ax.set_xlim(-3.5, 3.5)
    if n_pages > 1:
        ax.set_title(f"{title} (page {page}/{n_pages})")
    else:
        ax.set_title(title)
    ax.grid(True, axis="x", alpha=0.25)
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
    plot_corr_matrix(artifact, ax=ax, title="Correlation matrix", order="group_base", show_colorbar=True)
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
    ax.barh(y, impacts, color="#1D4ED8", alpha=0.85)
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

    ax.grid(True, axis="x", alpha=0.25)
    ax.grid(False, axis="y")
    return fig


def render_report(
    input_dir: Path | str,
    *,
    pdf: Path | str,
    svg_dir: Path | str | None,
    corr_include: str | None = None,
    corr_exclude: str | None = None,
    corr_top_n: int | None = None,
) -> None:
    _require_matplotlib()
    _apply_pub_style()

    from matplotlib.backends.backend_pdf import PdfPages
    import matplotlib.pyplot as plt

    input_dir = Path(input_dir).resolve()
    pdf = Path(pdf).resolve()
    if svg_dir is not None:
        svg_dir = Path(svg_dir).resolve()

    targets = _SaveTargets(pdf=pdf, svg_dir=svg_dir)

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
            fig, ax = plt.subplots(figsize=(7.6, 2.8))
            ax.text(0.5, 0.5, f"No report artifacts found in: {input_dir}", ha="center", va="center")
            ax.axis("off")
            save(fig, "empty_report")
            plt.close(fig)


def _cmd_render(args: argparse.Namespace) -> int:
    render_report(
        Path(args.input_dir),
        pdf=Path(args.pdf),
        svg_dir=Path(args.svg_dir) if args.svg_dir else None,
        corr_include=args.corr_include,
        corr_exclude=args.corr_exclude,
        corr_top_n=int(args.corr_top_n) if args.corr_top_n is not None else None,
    )
    return 0


def main(argv: Sequence[str] | None = None) -> int:
    p = argparse.ArgumentParser(prog="python -m nextstat.report")
    sub = p.add_subparsers(dest="cmd", required=True)

    r = sub.add_parser("render", help="Render report PDF (+ optional per-plot SVG) from artifacts directory")
    r.add_argument("--input-dir", required=True, help="Directory containing artifacts JSON files")
    r.add_argument("--pdf", required=True, help="Output PDF path")
    r.add_argument("--svg-dir", default=None, help="Optional directory for per-plot SVGs")
    r.add_argument("--corr-include", default=None, help="Regex: include parameters for corr plot")
    r.add_argument("--corr-exclude", default=None, help="Regex: exclude parameters for corr plot")
    r.add_argument("--corr-top-n", default=None, help="Keep top-N parameters by max |corr| (after filters)")
    r.set_defaults(fn=_cmd_render)

    args = p.parse_args(list(argv) if argv is not None else None)
    return int(args.fn(args))


if __name__ == "__main__":  # pragma: no cover
    raise SystemExit(main())
