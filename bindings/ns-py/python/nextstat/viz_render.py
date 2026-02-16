"""Single-artifact renderer for NextStat viz outputs.

This module renders one artifact JSON (pulls/corr/ranking) directly to an image
or vector file (PNG/SVG/PDF), so users can go from `nextstat viz ...` output to
a plot in one step.
"""

from __future__ import annotations

import argparse
import json
from pathlib import Path
from typing import Any, Mapping


def _require_matplotlib():
    try:
        import matplotlib as mpl
        mpl.use("Agg", force=True)
        import matplotlib.pyplot as plt  # noqa: F401
    except Exception as e:  # pragma: no cover
        raise ImportError("Missing dependency: matplotlib. Install via `pip install nextstat[viz]`.") from e


def _read_json(path: Path) -> Mapping[str, Any]:
    return json.loads(path.read_text())


def render_artifact(
    *,
    kind: str,
    artifact: Mapping[str, Any],
    output: Path,
    title: str | None = None,
    dpi: int = 220,
    corr_include: str | None = None,
    corr_exclude: str | None = None,
    corr_top_n: int | None = None,
) -> None:
    _require_matplotlib()
    import matplotlib.pyplot as plt

    from . import viz

    k = str(kind).strip().lower()
    if k == "pulls":
        ax = viz.plot_pulls(artifact, title=title or "Pulls + constraints")
        fig = ax.figure
    elif k == "corr":
        ax = viz.plot_corr_matrix(
            artifact,
            title=title or "Correlation matrix",
            include=corr_include,
            exclude=corr_exclude,
            top_n=corr_top_n,
        )
        fig = ax.figure
    elif k == "ranking":
        ax_impact, _ax_pull = viz.plot_ranking(artifact, title=title or "Nuisance parameter ranking")
        fig = ax_impact.figure
    else:
        raise ValueError(f"unsupported kind: {kind!r} (expected: pulls|corr|ranking)")

    output.parent.mkdir(parents=True, exist_ok=True)
    fig.savefig(output, dpi=max(72, int(dpi)))
    plt.close(fig)


def _cmd_render(args: argparse.Namespace) -> None:
    artifact = _read_json(Path(args.input))
    render_artifact(
        kind=str(args.kind),
        artifact=artifact,
        output=Path(args.output),
        title=args.title,
        dpi=int(args.dpi),
        corr_include=args.corr_include,
        corr_exclude=args.corr_exclude,
        corr_top_n=int(args.corr_top_n) if args.corr_top_n is not None else None,
    )


def _build_parser() -> argparse.ArgumentParser:
    p = argparse.ArgumentParser(prog="python -m nextstat.viz_render")
    sub = p.add_subparsers(dest="command", required=True)

    r = sub.add_parser("render", help="Render one viz artifact (pulls/corr/ranking) to a file.")
    r.add_argument("--kind", required=True, choices=["pulls", "corr", "ranking"])
    r.add_argument("--input", required=True, help="Input artifact JSON path.")
    r.add_argument("--output", required=True, help="Output image/document path (.png/.svg/.pdf).")
    r.add_argument("--title", default=None, help="Optional title override.")
    r.add_argument("--dpi", default="220", help="Output DPI (for raster formats).")
    r.add_argument("--corr-include", default=None, help="corr-only include regex.")
    r.add_argument("--corr-exclude", default=None, help="corr-only exclude regex.")
    r.add_argument("--corr-top-n", default=None, help="corr-only top-N by max |corr|.")
    r.set_defaults(func=_cmd_render)
    return p


def main(argv: list[str] | None = None) -> None:
    p = _build_parser()
    args = p.parse_args(argv)
    args.func(args)


if __name__ == "__main__":  # pragma: no cover
    main()
