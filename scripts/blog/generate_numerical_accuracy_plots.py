#!/usr/bin/env python3
"""Generate SVG plots for the numerical-accuracy blog post.

This script is dependency-free (stdlib only). It reads the snapshot scan
artifacts under:

  docs/blog/assets/numerical-accuracy/data/<fixture>/
    - root_profile_scan.json
    - pyhf_profile_scan.json
    - nextstat_profile_scan.json
    - summary.json

and writes SVGs to:

  docs/blog/assets/numerical-accuracy/
"""

from __future__ import annotations

import json
import math
from dataclasses import dataclass
from pathlib import Path


REPO_ROOT = Path(__file__).resolve().parents[2]
DATA_ROOT = REPO_ROOT / "docs" / "blog" / "assets" / "numerical-accuracy" / "data"
OUT_ROOT = REPO_ROOT / "docs" / "blog" / "assets" / "numerical-accuracy"


def _load_json(path: Path) -> dict:
    return json.loads(path.read_text())


def _nearest_point(points: list[dict], mu: float) -> dict:
    best = min(points, key=lambda p: abs(float(p.get("mu")) - mu))
    if abs(float(best.get("mu")) - mu) > 1e-8:
        raise RuntimeError(f"no matching mu point found: mu={mu} best_mu={best.get('mu')}")
    return best


@dataclass(frozen=True)
class Scan:
    mu: list[float]
    q_mu: list[float]
    nll_mu: list[float] | None


def _read_scan(path: Path) -> Scan:
    obj = _load_json(path)
    points = list(obj.get("points") or [])
    mu = [float(p["mu"]) for p in points]
    q_mu = [float(p["q_mu"]) for p in points]
    if all("nll_mu" in p for p in points):
        nll_mu = [float(p["nll_mu"]) for p in points]
    else:
        nll_mu = None
    return Scan(mu=mu, q_mu=q_mu, nll_mu=nll_mu)


@dataclass(frozen=True)
class FixtureScans:
    mu: list[float]
    root: Scan
    pyhf: Scan
    nextstat: Scan


def _read_fixture(fixture: str) -> FixtureScans:
    base = DATA_ROOT / fixture
    root = _read_scan(base / "root_profile_scan.json")
    pyhf = _read_scan(base / "pyhf_profile_scan.json")
    nextstat = _read_scan(base / "nextstat_profile_scan.json")

    # Use NextStat mu-grid as the canonical grid (it includes NLL values).
    mu = list(nextstat.mu)

    def remap(scan: Scan) -> Scan:
        points = [{"mu": m, "q_mu": q} for m, q in zip(scan.mu, scan.q_mu)]
        if scan.nll_mu is not None:
            for i, nll in enumerate(scan.nll_mu):
                points[i]["nll_mu"] = nll

        mu_out: list[float] = []
        q_out: list[float] = []
        nll_out: list[float] | None = [] if scan.nll_mu is not None else None
        for m in mu:
            p = _nearest_point(points, float(m))
            mu_out.append(float(m))
            q_out.append(float(p["q_mu"]))
            if nll_out is not None:
                nll_out.append(float(p["nll_mu"]))
        return Scan(mu=mu_out, q_mu=q_out, nll_mu=nll_out)

    return FixtureScans(mu=mu, root=remap(root), pyhf=remap(pyhf), nextstat=remap(nextstat))


def _nice_step(raw_step: float) -> float:
    if raw_step <= 0:
        return 1.0
    exp = math.floor(math.log10(raw_step))
    frac = raw_step / (10**exp)
    if frac <= 1:
        nice = 1
    elif frac <= 2:
        nice = 2
    elif frac <= 5:
        nice = 5
    else:
        nice = 10
    return nice * (10**exp)


def _nice_ticks(vmin: float, vmax: float, *, target: int = 6) -> list[float]:
    if not math.isfinite(vmin) or not math.isfinite(vmax):
        return [0.0]
    if vmin == vmax:
        if vmin == 0.0:
            return [-1.0, 0.0, 1.0]
        pad = abs(vmin) * 0.1
        vmin -= pad
        vmax += pad

    span = vmax - vmin
    step = _nice_step(span / max(target - 1, 1))
    start = math.floor(vmin / step) * step
    stop = math.ceil(vmax / step) * step

    ticks: list[float] = []
    v = start
    # Add a small epsilon to guard against float drift.
    eps = step * 1e-9
    while v <= stop + eps:
        ticks.append(v)
        v += step
    return ticks


def _fmt_tick(v: float) -> str:
    if v == 0:
        return "0"
    av = abs(v)
    if av >= 1e4 or av < 1e-3:
        return f"{v:.2e}"
    if av < 1:
        return f"{v:.4f}".rstrip("0").rstrip(".")
    if av < 10:
        return f"{v:.3f}".rstrip("0").rstrip(".")
    if av < 100:
        return f"{v:.2f}".rstrip("0").rstrip(".")
    return f"{v:.1f}".rstrip("0").rstrip(".")


def _polyline_points(xs: list[float], ys: list[float]) -> str:
    return " ".join(f"{x:.3f},{y:.3f}" for x, y in zip(xs, ys))


@dataclass(frozen=True)
class Series:
    name: str
    y: list[float]
    color: str
    width: float = 2.5
    dash: str | None = None


def _svg_line_chart(
    *,
    title: str,
    x_label: str,
    y_label: str,
    x: list[float],
    series: list[Series],
    out_path: Path,
    y_floor: float | None = None,
    width: int = 960,
    height: int = 560,
) -> None:
    if not x:
        raise RuntimeError("empty x")
    if any(len(s.y) != len(x) for s in series):
        raise RuntimeError("series length mismatch")

    x_min = min(x)
    x_max = max(x)

    y_vals = [v for s in series for v in s.y if math.isfinite(v)]
    if not y_vals:
        y_min, y_max = -1.0, 1.0
    else:
        y_min = min(y_vals)
        y_max = max(y_vals)

    if y_min == y_max:
        pad = 1.0 if y_min == 0 else abs(y_min) * 0.1
        y_min -= pad
        y_max += pad

    # Symmetric bounds around 0 for residual-style plots.
    if y_min < 0 < y_max:
        max_abs = max(abs(y_min), abs(y_max))
        y_min, y_max = -max_abs, max_abs

    y_pad = (y_max - y_min) * 0.06
    y_min -= y_pad
    y_max += y_pad

    if y_floor is not None:
        y_min = max(y_min, float(y_floor))
        if y_min == y_max:
            y_max = y_min + 1.0

    margin_left = 92
    margin_right = 28
    margin_top = 72
    margin_bottom = 78

    plot_w = width - margin_left - margin_right
    plot_h = height - margin_top - margin_bottom

    def sx(v: float) -> float:
        if x_max == x_min:
            return margin_left + plot_w / 2
        return margin_left + (v - x_min) / (x_max - x_min) * plot_w

    def sy(v: float) -> float:
        if y_max == y_min:
            return margin_top + plot_h / 2
        return margin_top + (1.0 - (v - y_min) / (y_max - y_min)) * plot_h

    x_ticks = [0.0, 0.5, 1.0, 1.5, 2.0, 2.5, 3.0] if (x_min <= 0 and x_max >= 3) else _nice_ticks(x_min, x_max)
    y_ticks = _nice_ticks(y_min, y_max)

    # Precompute polyline paths.
    lines = []
    for s in series:
        px = [sx(v) for v in x]
        py = [sy(v) for v in s.y]
        lines.append((s, px, py))

    # Legend layout.
    legend_x = margin_left + plot_w - 220
    legend_y = margin_top - 38
    legend_line_w = 26
    legend_row_h = 18

    def esc(text: str) -> str:
        return (
            text.replace("&", "&amp;")
            .replace("<", "&lt;")
            .replace(">", "&gt;")
            .replace('"', "&quot;")
            .replace("'", "&apos;")
        )

    svg_parts: list[str] = []
    svg_parts.append(
        f'<svg xmlns="http://www.w3.org/2000/svg" width="{width}" height="{height}" viewBox="0 0 {width} {height}">'
    )
    svg_parts.append(
        "<style>"
        "text{font-family:ui-sans-serif,system-ui,-apple-system,Segoe UI,Roboto,Helvetica,Arial,sans-serif;}"
        ".title{font-size:18px;font-weight:600;fill:#111827;}"
        ".label{font-size:13px;fill:#111827;}"
        ".tick{font-size:12px;fill:#374151;}"
        ".grid{stroke:#e5e7eb;stroke-width:1;}"
        ".axis{stroke:#111827;stroke-width:1.2;}"
        "</style>"
    )

    # Background.
    svg_parts.append(f'<rect x="0" y="0" width="{width}" height="{height}" fill="#ffffff" />')

    # Title.
    svg_parts.append(f'<text x="{margin_left}" y="34" class="title">{esc(title)}</text>')

    # Grid + ticks (Y).
    for t in y_ticks:
        y = sy(t)
        svg_parts.append(
            f'<line x1="{margin_left}" y1="{y:.3f}" x2="{margin_left + plot_w}" y2="{y:.3f}" class="grid" />'
        )
        svg_parts.append(
            f'<text x="{margin_left - 12}" y="{y + 4:.3f}" text-anchor="end" class="tick">{esc(_fmt_tick(t))}</text>'
        )

    # Grid + ticks (X).
    for t in x_ticks:
        if t < x_min - 1e-9 or t > x_max + 1e-9:
            continue
        x_px = sx(t)
        svg_parts.append(
            f'<line x1="{x_px:.3f}" y1="{margin_top}" x2="{x_px:.3f}" y2="{margin_top + plot_h}" class="grid" />'
        )
        svg_parts.append(
            f'<text x="{x_px:.3f}" y="{margin_top + plot_h + 22}" text-anchor="middle" class="tick">{esc(_fmt_tick(t))}</text>'
        )

    # Axes.
    svg_parts.append(
        f'<line x1="{margin_left}" y1="{margin_top}" x2="{margin_left}" y2="{margin_top + plot_h}" class="axis" />'
    )
    svg_parts.append(
        f'<line x1="{margin_left}" y1="{margin_top + plot_h}" x2="{margin_left + plot_w}" y2="{margin_top + plot_h}" class="axis" />'
    )

    # Emphasize y=0 line if it is inside the plotting range.
    if y_min < 0 < y_max:
        y0 = sy(0.0)
        svg_parts.append(
            f'<line x1="{margin_left}" y1="{y0:.3f}" x2="{margin_left + plot_w}" y2="{y0:.3f}" stroke="#9ca3af" stroke-width="1.2" />'
        )

    # Labels.
    svg_parts.append(
        f'<text x="{margin_left + plot_w / 2:.3f}" y="{height - 18}" text-anchor="middle" class="label">{esc(x_label)}</text>'
    )
    svg_parts.append(
        f'<text x="22" y="{margin_top + plot_h / 2:.3f}" text-anchor="middle" class="label" transform="rotate(-90 22 {margin_top + plot_h / 2:.3f})">{esc(y_label)}</text>'
    )

    # Lines.
    for s, px, py in lines:
        dash = f' stroke-dasharray="{s.dash}"' if s.dash else ""
        svg_parts.append(
            f'<polyline fill="none" stroke="{s.color}" stroke-width="{s.width}"{dash} points="{_polyline_points(px, py)}" />'
        )

    # Legend.
    svg_parts.append(
        f'<rect x="{legend_x}" y="{legend_y}" width="212" height="{legend_row_h * len(series) + 10}" fill="#ffffff" stroke="#e5e7eb" />'
    )
    for i, s in enumerate(series):
        y = legend_y + 22 + i * legend_row_h
        dash = f' stroke-dasharray="{s.dash}"' if s.dash else ""
        svg_parts.append(
            f'<line x1="{legend_x + 12}" y1="{y}" x2="{legend_x + 12 + legend_line_w}" y2="{y}" stroke="{s.color}" stroke-width="{s.width}"{dash} />'
        )
        svg_parts.append(f'<text x="{legend_x + 12 + legend_line_w + 10}" y="{y + 4}" class="tick">{esc(s.name)}</text>')

    svg_parts.append("</svg>")
    out_path.write_text("\n".join(svg_parts))


def _write_all() -> None:
    OUT_ROOT.mkdir(parents=True, exist_ok=True)

    multichannel = _read_fixture("multichannel")
    xmlimport = _read_fixture("xmlimport")
    coupled = _read_fixture("coupled_histosys")

    # 1) ShapeSys: residuals (ROOT - NextStat and NextStat - pyhf)
    mult_d_root_ns = [qr - qn for qr, qn in zip(multichannel.root.q_mu, multichannel.nextstat.q_mu)]
    mult_d_ns_pyhf = [qn - qp for qn, qp in zip(multichannel.nextstat.q_mu, multichannel.pyhf.q_mu)]
    _svg_line_chart(
        title="multichannel (ShapeSys): Δq(mu) residuals",
        x_label="mu",
        y_label="Δq(mu)",
        x=multichannel.mu,
        series=[
            Series(name="ROOT - NextStat", y=mult_d_root_ns, color="#d62728"),
            Series(name="NextStat - pyhf", y=mult_d_ns_pyhf, color="#6f42c1"),
        ],
        out_path=OUT_ROOT / "multichannel-deltaq.svg",
    )

    # 2) OverallSys: ROOT bias (ROOT - NextStat and ROOT - pyhf)
    xml_d_root_ns = [qr - qn for qr, qn in zip(xmlimport.root.q_mu, xmlimport.nextstat.q_mu)]
    xml_d_root_pyhf = [qr - qp for qr, qp in zip(xmlimport.root.q_mu, xmlimport.pyhf.q_mu)]
    _svg_line_chart(
        title="xmlimport (OverallSys): ROOT bias in q(mu)",
        x_label="mu",
        y_label="Δq(mu)",
        x=xmlimport.mu,
        series=[
            Series(name="ROOT - NextStat", y=xml_d_root_ns, color="#d62728"),
            Series(name="ROOT - pyhf", y=xml_d_root_pyhf, color="#ff7f0e"),
        ],
        y_floor=0.0,
        out_path=OUT_ROOT / "xmlimport-deltaq.svg",
    )

    # 3) Coupled HistoSys: q(mu) curves
    _svg_line_chart(
        title="coupled_histosys: q(mu) profile scan",
        x_label="mu",
        y_label="q(mu)",
        x=coupled.mu,
        series=[
            Series(name="ROOT", y=coupled.root.q_mu, color="#d62728"),
            Series(name="pyhf", y=coupled.pyhf.q_mu, color="#1f77b4"),
            Series(name="NextStat", y=coupled.nextstat.q_mu, color="#2ca02c", dash="5 4", width=2.0),
        ],
        y_floor=0.0,
        out_path=OUT_ROOT / "coupled-histosys-qmu.svg",
    )

    # 4) Coupled HistoSys: NLL offset (ROOT - NextStat), showing non-constant offset.
    if coupled.root.nll_mu is None or coupled.nextstat.nll_mu is None:
        raise RuntimeError("expected per-mu NLL in root + nextstat scans for coupled_histosys")
    offset = [nr - nn for nr, nn in zip(coupled.root.nll_mu, coupled.nextstat.nll_mu)]
    _svg_line_chart(
        title="coupled_histosys: NLL offset (ROOT - NextStat)",
        x_label="mu",
        y_label="NLL offset",
        x=coupled.mu,
        series=[Series(name="offset", y=offset, color="#111827")],
        out_path=OUT_ROOT / "coupled-histosys-nll-offset.svg",
    )


def main(argv: list[str] | None = None) -> int:
    _ = argv  # reserved for future flags
    for fixture in ("xmlimport", "multichannel", "coupled_histosys"):
        if not (DATA_ROOT / fixture).exists():
            raise RuntimeError(f"missing fixture data directory: {DATA_ROOT / fixture}")
    _write_all()
    print(f"Wrote plots under: {OUT_ROOT}")
    return 0


if __name__ == "__main__":
    raise SystemExit(main())
