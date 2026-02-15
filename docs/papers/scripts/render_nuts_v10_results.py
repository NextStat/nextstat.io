#!/usr/bin/env python3
"""Render artifact-driven v10 NUTS results into the paper + lightweight SVG figures.

Inputs:
- `benchmarks/nextstat-public-benchmarks/suites/bayesian/results_v10/` (checked-in artifacts)

Outputs:
- Updates `docs/papers/nuts-progressive-sampling.md` between AUTOGEN markers
- Writes SVG figures under `docs/papers/assets/nuts-v10/`

No external plotting deps required (generates SVG directly).
"""

from __future__ import annotations

import argparse
import json
import math
import statistics
from dataclasses import dataclass
from pathlib import Path


@dataclass(frozen=True)
class RunRow:
    seed_dir: str
    seed: int
    case_id: str
    backend: str
    wall_time_s: float | None
    min_ess_bulk: float | None
    min_ess_bulk_per_sec: float | None
    max_r_hat: float | None
    divergence_rate: float | None
    max_treedepth_rate: float | None
    min_ebfmi: float | None
    n_warmup: int | None
    n_samples: int | None
    n_chains: int | None
    dataset_seed: int | None
    target_accept: float | None
    metric: str | None


CASE_LABEL = {
    "glm_logistic_regression": "GLM logistic (6p)",
    "hier_random_intercept_non_centered": "Hierarchical logistic RI (non-centered, 22p)",
    "eight_schools_non_centered": "Eight Schools (non-centered, 10p)",
    "histfactory_simple_8p": "HistFactory simple (8p)",
}


def _safe_float(x) -> float | None:
    try:
        v = float(x)
    except Exception:
        return None
    if not math.isfinite(v):
        return None
    return v


def _safe_int(x) -> int | None:
    try:
        return int(x)
    except Exception:
        return None


def _mean_std(vals: list[float]) -> tuple[float | None, float | None]:
    if not vals:
        return None, None
    if len(vals) == 1:
        return vals[0], 0.0
    return statistics.mean(vals), statistics.stdev(vals)


def _fmt(x: float | None, *, digits: int = 3) -> str:
    if x is None:
        return "—"
    if abs(x) >= 1000:
        return f"{x:,.0f}"
    if abs(x) >= 100:
        return f"{x:,.1f}"
    if abs(x) >= 10:
        return f"{x:,.2f}"
    return f"{x:.{digits}f}".rstrip("0").rstrip(".")


def _fmt_pm(mean: float | None, std: float | None) -> str:
    if mean is None:
        return "—"
    if std is None:
        return _fmt(mean)
    if std == 0.0:
        return _fmt(mean)
    return f"{_fmt(mean)} ± {_fmt(std)}"


def _read_json(path: Path) -> dict:
    return json.loads(path.read_text())


def _iter_seed_dirs(results_dir: Path) -> list[tuple[int, Path]]:
    out: list[tuple[int, Path]] = []
    for d in sorted(results_dir.glob("seed_*")):
        if not d.is_dir():
            continue
        tok = d.name.split("_", 1)[-1]
        try:
            seed = int(tok)
        except Exception:
            continue
        out.append((seed, d))
    return out


def _collect_rows(results_dir: Path) -> list[RunRow]:
    rows: list[RunRow] = []
    for seed, seed_dir in _iter_seed_dirs(results_dir):
        suite_path = seed_dir / "bayesian_suite.json"
        if not suite_path.exists():
            continue
        suite_obj = _read_json(suite_path)
        cases = suite_obj.get("cases") if isinstance(suite_obj.get("cases"), list) else []
        for c in cases:
            rel = c.get("path")
            if not isinstance(rel, str) or not rel:
                continue
            case_path = (seed_dir / rel).resolve()
            if not case_path.exists():
                continue
            obj = _read_json(case_path)
            cfg = obj.get("config") if isinstance(obj.get("config"), dict) else {}
            diag_sum = obj.get("diagnostics_summary") if isinstance(obj.get("diagnostics_summary"), dict) else {}

            rows.append(
                RunRow(
                    seed_dir=seed_dir.name,
                    seed=seed,
                    case_id=str(obj.get("case") or c.get("case") or "unknown"),
                    backend=str(obj.get("backend") or c.get("backend") or "unknown"),
                    wall_time_s=_safe_float(obj.get("timing", {}).get("wall_time_s")) if isinstance(obj.get("timing"), dict) else _safe_float(c.get("wall_time_s")),
                    min_ess_bulk=_safe_float(diag_sum.get("min_ess_bulk")) if diag_sum else _safe_float(c.get("min_ess_bulk")),
                    min_ess_bulk_per_sec=_safe_float(obj.get("timing", {}).get("ess_bulk_per_sec", {}).get("min")) if isinstance(obj.get("timing"), dict) else _safe_float(c.get("min_ess_bulk_per_sec")),
                    max_r_hat=_safe_float(diag_sum.get("max_r_hat")) if diag_sum else _safe_float(c.get("max_r_hat")),
                    divergence_rate=_safe_float(diag_sum.get("divergence_rate")) if diag_sum else None,
                    max_treedepth_rate=_safe_float(diag_sum.get("max_treedepth_rate")) if diag_sum else None,
                    min_ebfmi=_safe_float(diag_sum.get("min_ebfmi")) if diag_sum else None,
                    n_warmup=_safe_int(cfg.get("n_warmup")),
                    n_samples=_safe_int(cfg.get("n_samples")),
                    n_chains=_safe_int(cfg.get("n_chains")),
                    dataset_seed=_safe_int(cfg.get("dataset_seed")),
                    target_accept=_safe_float(cfg.get("target_accept")),
                    metric=str(cfg.get("metric")) if isinstance(cfg.get("metric"), str) else None,
                )
            )
    return rows


def _svg_header(width: int, height: int) -> list[str]:
    return [
        '<?xml version="1.0" encoding="UTF-8"?>',
        f'<svg xmlns="http://www.w3.org/2000/svg" width="{width}" height="{height}" viewBox="0 0 {width} {height}">',
        "<style>",
        "  text { font-family: ui-sans-serif, system-ui, -apple-system, Segoe UI, Roboto, Helvetica, Arial; fill: #111827; }",
        "  .muted { fill: #6b7280; }",
        "  .label { font-size: 14px; }",
        "  .title { font-size: 16px; font-weight: 700; }",
        "  .small { font-size: 12px; }",
        "</style>",
    ]


def _svg_footer() -> list[str]:
    return ["</svg>"]


def _write_svg(path: Path, lines: list[str]) -> None:
    path.parent.mkdir(parents=True, exist_ok=True)
    path.write_text("\n".join(lines) + "\n")


def _render_bar_svg(*, out_path: Path, cases: list[str], ns_vals: list[float], st_vals: list[float]) -> None:
    width = 900
    height = 320
    pad_l = 240
    pad_r = 40
    pad_t = 40
    pad_b = 40
    chart_w = width - pad_l - pad_r
    chart_h = height - pad_t - pad_b

    max_v = max(ns_vals + st_vals + [1.0])
    max_v *= 1.05

    n = len(cases)
    row_h = chart_h / max(n, 1)
    bar_h = row_h * 0.28
    gap = row_h * 0.12

    c_ns = "#111827"
    c_st = "#9ca3af"

    svg = _svg_header(width, height)
    svg.append(f'<text x="{pad_l}" y="24" class="title">ESS/sec (min bulk) — NextStat vs CmdStan</text>')
    svg.append(f'<text x="{pad_l}" y="38" class="small muted">Artifact-driven from results_v10 (avg across seeds)</text>')

    # Axes baseline
    x0 = pad_l
    y0 = pad_t + chart_h
    svg.append(f'<line x1="{x0}" y1="{y0}" x2="{x0 + chart_w}" y2="{y0}" stroke="#e5e7eb" />')

    # Ticks (0, 0.5, 1.0) scaled to max
    for frac in [0.0, 0.25, 0.5, 0.75, 1.0]:
        v = frac * max_v
        x = x0 + (v / max_v) * chart_w
        svg.append(f'<line x1="{x:.1f}" y1="{pad_t}" x2="{x:.1f}" y2="{y0}" stroke="#f3f4f6" />')
        svg.append(f'<text x="{x:.1f}" y="{y0 + 16}" class="small muted" text-anchor="middle">{_fmt(v)}</text>')

    for i, case in enumerate(cases):
        cy = pad_t + (i + 0.5) * row_h
        label = CASE_LABEL.get(case, case)
        svg.append(f'<text x="{pad_l - 12}" y="{cy + 5:.1f}" class="label" text-anchor="end">{label}</text>')

        ns = ns_vals[i]
        st = st_vals[i]
        # Two stacked bars per row
        y_ns = cy - bar_h - gap / 2
        y_st = cy + gap / 2
        w_ns = (ns / max_v) * chart_w
        w_st = (st / max_v) * chart_w
        svg.append(f'<rect x="{x0}" y="{y_ns:.1f}" width="{w_ns:.1f}" height="{bar_h:.1f}" fill="{c_ns}" rx="2" />')
        svg.append(f'<rect x="{x0}" y="{y_st:.1f}" width="{w_st:.1f}" height="{bar_h:.1f}" fill="{c_st}" rx="2" />')

        ratio = ns / st if st > 0 else float("nan")
        svg.append(
            f'<text x="{x0 + chart_w + 8}" y="{cy + 4:.1f}" class="small muted">'
            f'{_fmt(ratio, digits=2)}x</text>'
        )

    # Legend
    lx = pad_l
    ly = height - 10
    svg.append(f'<rect x="{lx}" y="{ly - 10}" width="12" height="12" fill="{c_ns}" rx="2" />')
    svg.append(f'<text x="{lx + 18}" y="{ly}" class="small muted">NextStat</text>')
    svg.append(f'<rect x="{lx + 90}" y="{ly - 10}" width="12" height="12" fill="{c_st}" rx="2" />')
    svg.append(f'<text x="{lx + 108}" y="{ly}" class="small muted">CmdStan</text>')

    svg.extend(_svg_footer())
    _write_svg(out_path, svg)


def _render_seed_scatter_svg(*, out_path: Path, rows: list[RunRow], case_ids: list[str]) -> None:
    # Scatter of per-seed ESS/sec for both backends.
    width = 900
    height = 360
    pad_l = 240
    pad_r = 40
    pad_t = 40
    pad_b = 40
    chart_w = width - pad_l - pad_r
    chart_h = height - pad_t - pad_b

    # Gather values
    pts: list[tuple[int, float, str, str]] = []  # (case_i, value, backend, seed_dir)
    for ci, case in enumerate(case_ids):
        for r in rows:
            if r.case_id != case:
                continue
            if r.min_ess_bulk_per_sec is None:
                continue
            if r.backend not in ("nextstat", "cmdstanpy"):
                continue
            pts.append((ci, float(r.min_ess_bulk_per_sec), r.backend, r.seed_dir))

    max_v = max([v for _, v, _, _ in pts] + [1.0])
    max_v *= 1.05

    n = len(case_ids)
    row_h = chart_h / max(n, 1)

    c_ns = "#111827"
    c_st = "#9ca3af"

    svg = _svg_header(width, height)
    svg.append(f'<text x="{pad_l}" y="24" class="title">Per-seed ESS/sec scatter (min bulk)</text>')
    svg.append(f'<text x="{pad_l}" y="38" class="small muted">Each dot = one seed run; x-axis is ESS/sec</text>')

    x0 = pad_l
    y0 = pad_t + chart_h
    svg.append(f'<line x1="{x0}" y1="{y0}" x2="{x0 + chart_w}" y2="{y0}" stroke="#e5e7eb" />')

    for frac in [0.0, 0.25, 0.5, 0.75, 1.0]:
        v = frac * max_v
        x = x0 + (v / max_v) * chart_w
        svg.append(f'<line x1="{x:.1f}" y1="{pad_t}" x2="{x:.1f}" y2="{y0}" stroke="#f3f4f6" />')
        svg.append(f'<text x="{x:.1f}" y="{y0 + 16}" class="small muted" text-anchor="middle">{_fmt(v)}</text>')

    # Slight jitter per backend to avoid overlap
    for ci, case in enumerate(case_ids):
        cy = pad_t + (ci + 0.5) * row_h
        label = CASE_LABEL.get(case, case)
        svg.append(f'<text x="{pad_l - 12}" y="{cy + 5:.1f}" class="label" text-anchor="end">{label}</text>')

        local = [p for p in pts if p[0] == ci]
        # deterministic order: cmdstan then nextstat
        local.sort(key=lambda t: 0 if t[2] == "cmdstanpy" else 1)
        for _, val, backend, seed_dir in local:
            x = x0 + (val / max_v) * chart_w
            y = cy + (-8 if backend == "nextstat" else 8)
            color = c_ns if backend == "nextstat" else c_st
            svg.append(f'<circle cx="{x:.1f}" cy="{y:.1f}" r="4.2" fill="{color}" />')
            svg.append(f'<text x="{x + 8:.1f}" y="{y + 4:.1f}" class="small muted">{seed_dir}</text>')

    # Legend
    lx = pad_l
    ly = height - 10
    svg.append(f'<circle cx="{lx + 6}" cy="{ly - 4}" r="5" fill="{c_ns}" />')
    svg.append(f'<text x="{lx + 18}" y="{ly}" class="small muted">NextStat</text>')
    svg.append(f'<circle cx="{lx + 96}" cy="{ly - 4}" r="5" fill="{c_st}" />')
    svg.append(f'<text x="{lx + 108}" y="{ly}" class="small muted">CmdStan</text>')

    svg.extend(_svg_footer())
    _write_svg(out_path, svg)


def _render_autogen_markdown(*, rows: list[RunRow], results_dir: Path) -> str:
    # Aggregate by (case, backend)
    by_key: dict[tuple[str, str], list[RunRow]] = {}
    for r in rows:
        if r.backend not in ("nextstat", "cmdstanpy"):
            continue
        by_key.setdefault((r.case_id, r.backend), []).append(r)

    case_ids = [c for c in CASE_LABEL.keys() if (c, "nextstat") in by_key and c != "histfactory_simple_8p"]
    # Keep deterministic ordering
    case_ids = [c for c in ["glm_logistic_regression", "hier_random_intercept_non_centered", "eight_schools_non_centered"] if c in case_ids]

    # Infer config consensus from NextStat rows
    ns_cfgs = [r for r in rows if r.backend == "nextstat" and r.case_id == "glm_logistic_regression"]
    n_chains = ns_cfgs[0].n_chains if ns_cfgs and ns_cfgs[0].n_chains is not None else None
    n_warmup = ns_cfgs[0].n_warmup if ns_cfgs and ns_cfgs[0].n_warmup is not None else None
    n_samples = ns_cfgs[0].n_samples if ns_cfgs and ns_cfgs[0].n_samples is not None else None
    dataset_seed = ns_cfgs[0].dataset_seed if ns_cfgs and ns_cfgs[0].dataset_seed is not None else None

    seeds = sorted({r.seed for r in rows if r.backend == "nextstat"})

    lines: list[str] = []
    lines.append("<!-- AUTOGEN:V10_RESULTS_BEGIN -->")
    lines.append("")
    lines.append("This section is generated from checked-in benchmark artifacts:")
    lines.append("")
    # Prefer repo-relative paths in published docs.
    repo_root = Path(__file__).resolve().parents[3]
    try:
        rel = results_dir.relative_to(repo_root)
        lines.append(f"- Artifacts: `{rel.as_posix()}`")
    except Exception:
        lines.append(f"- Artifacts: `{results_dir.as_posix()}`")
    if dataset_seed is not None:
        lines.append(f"- Dataset seed: `{dataset_seed}` (fixed)")
    if seeds:
        lines.append(f"- Chain seeds: `{', '.join(map(str, seeds))}`")
    if n_chains is not None and n_warmup is not None and n_samples is not None:
        lines.append(f"- Config: `{n_chains}` chains, `{n_warmup}` warmup, `{n_samples}` samples, diagonal metric")
    lines.append("")
    lines.append("### 7.1 ESS/sec (avg across seeds)")
    lines.append("")
    lines.append("| Model | NextStat ESS/sec | CmdStan ESS/sec | Ratio |")
    lines.append("|---|---:|---:|---:|")
    esssec_ratio_by_case: dict[str, float | None] = {}
    for case in case_ids:
        ns = [r.min_ess_bulk_per_sec for r in by_key.get((case, "nextstat"), []) if r.min_ess_bulk_per_sec is not None]
        st = [r.min_ess_bulk_per_sec for r in by_key.get((case, "cmdstanpy"), []) if r.min_ess_bulk_per_sec is not None]
        m_ns, s_ns = _mean_std([float(x) for x in ns])
        m_st, s_st = _mean_std([float(x) for x in st])
        ratio = (m_ns / m_st) if (m_ns is not None and m_st is not None and m_st > 0) else None
        esssec_ratio_by_case[case] = ratio
        lines.append(
            f"| {CASE_LABEL.get(case, case)} | {_fmt_pm(m_ns, s_ns)} | {_fmt_pm(m_st, s_st)} | {_fmt(ratio, digits=2)}x |"
        )
    lines.append("")
    lines.append("Figures (generated):")
    lines.append("")
    lines.append(f"![ESS/sec bar chart](./assets/nuts-v10/ess_sec_bar.svg)")
    lines.append("")
    lines.append(f"![ESS/sec per-seed scatter](./assets/nuts-v10/ess_sec_seed_scatter.svg)")
    lines.append("")
    lines.append("### 7.2 Health gates (worst across seeds)")
    lines.append("")
    lines.append("| Model | Backend | Divergences | Max treedepth hits | Max R-hat | Min E-BFMI |")
    lines.append("|---|---|---:|---:|---:|---:|")
    for case in case_ids:
        for backend in ["nextstat", "cmdstanpy"]:
            rr = by_key.get((case, backend), [])
            div = [r.divergence_rate for r in rr if r.divergence_rate is not None]
            td = [r.max_treedepth_rate for r in rr if r.max_treedepth_rate is not None]
            rh = [r.max_r_hat for r in rr if r.max_r_hat is not None]
            eb = [r.min_ebfmi for r in rr if r.min_ebfmi is not None]
            lines.append(
                f"| {CASE_LABEL.get(case, case)} | {backend} | {_fmt(max(div) if div else None)} | {_fmt(max(td) if td else None)} | {_fmt(max(rh) if rh else None)} | {_fmt(min(eb) if eb else None)} |"
            )
    lines.append("")

    # Optional: derived ESS/leapfrog diagnostics (not part of v1 schemas).
    derived_path = results_dir / "derived_metrics.json"
    if derived_path.exists():
        try:
            d = _read_json(derived_path)
        except Exception:
            d = {}
        esslf = d.get("ess_per_leapfrog") if isinstance(d.get("ess_per_leapfrog"), dict) else {}
        cases_obj = esslf.get("cases") if isinstance(esslf.get("cases"), dict) else None
        # Back-compat: v1 stored a single case payload.
        v1_single = esslf if (cases_obj is None and isinstance(esslf.get("case"), str)) else None

        if cases_obj or v1_single:
            lines.append("### 7.3 ESS/leapfrog (algorithmic efficiency)")
            lines.append("")
            lines.append("This diagnostic isolates sampler efficiency per unit of Hamiltonian integration.")
            lines.append("")
            lines.append("| Case | NextStat | CmdStan | Ratio |")
            lines.append("|---|---:|---:|---:|")
            if cases_obj:
                esslf_ratio_by_case: dict[str, float | None] = {}
                for case in case_ids:
                    row = cases_obj.get(case) if isinstance(cases_obj.get(case), dict) else {}
                    ns_obj = row.get("nextstat") if isinstance(row.get("nextstat"), dict) else {}
                    st_obj = row.get("cmdstan") if isinstance(row.get("cmdstan"), dict) else {}
                    ns_mean = _safe_float(ns_obj.get("mean"))
                    ns_std = _safe_float(ns_obj.get("std"))
                    st_mean = _safe_float(st_obj.get("mean"))
                    st_std = _safe_float(st_obj.get("std"))
                    ratio = (ns_mean / st_mean) if (ns_mean is not None and st_mean is not None and st_mean > 0) else None
                    esslf_ratio_by_case[case] = ratio
                    lines.append(
                        f"| {CASE_LABEL.get(case, case)} | {_fmt_pm(ns_mean, ns_std)} | {_fmt_pm(st_mean, st_std)} | {_fmt(ratio, digits=2)}x |"
                    )
                lines.append("")
                method = esslf.get("ess_method")
                if isinstance(method, str) and method.strip():
                    lines.append(f"Note: ESS is computed via `{method.strip()}` (supplementary; not part of v1 public schemas).")
                    lines.append("Note: Leapfrog totals are summed over post-warmup draws (`n_leapfrog__` / `sample_stats.n_leapfrog`).")
                    lines.append("")

                # Optional: decompose wall-time ratio into algorithmic vs implied per-LF throughput.
                lines.append("### 7.4 ESS/sec decomposition (implied)")
                lines.append("")
                lines.append("Using the identity: (ESS/sec ratio) ≈ (ESS/LF ratio) × (LF/sec ratio).")
                lines.append("")
                lines.append("| Case | ESS/sec ratio | ESS/LF ratio | Implied LF/sec ratio |")
                lines.append("|---|---:|---:|---:|")
                for case in case_ids:
                    r_sec = esssec_ratio_by_case.get(case)
                    r_lf = esslf_ratio_by_case.get(case)
                    implied = (r_sec / r_lf) if (r_sec is not None and r_lf is not None and r_lf != 0.0) else None
                    lines.append(
                        f"| {CASE_LABEL.get(case, case)} | {_fmt(r_sec, digits=2)}x | {_fmt(r_lf, digits=2)}x | {_fmt(implied, digits=2)}x |"
                    )
                lines.append("")
            else:
                case = str(v1_single.get("case") or "glm_logistic_regression")  # type: ignore[union-attr]
                ns = _safe_float(v1_single.get("nextstat"))  # type: ignore[union-attr]
                st = _safe_float(v1_single.get("cmdstan"))  # type: ignore[union-attr]
                ratio = (ns / st) if (ns is not None and st is not None and st > 0) else None
                lines.append(f"| {CASE_LABEL.get(case, case)} | {_fmt(ns)} | {_fmt(st)} | {_fmt(ratio, digits=2)}x |")
                lines.append("")
                note = v1_single.get("note")  # type: ignore[union-attr]
                if isinstance(note, str) and note.strip():
                    lines.append(f"Note: {note.strip()}")
                    lines.append("")

    lines.append("<!-- AUTOGEN:V10_RESULTS_END -->")
    return "\n".join(lines) + "\n"


def _replace_autogen_block(*, paper_path: Path, new_block: str) -> None:
    s = paper_path.read_text()
    begin = "<!-- AUTOGEN:V10_RESULTS_BEGIN -->"
    end = "<!-- AUTOGEN:V10_RESULTS_END -->"
    i = s.find(begin)
    j = s.find(end)
    if i == -1 or j == -1 or j < i:
        raise SystemExit(f"autogen markers not found in {paper_path}")
    j_end = j + len(end)
    out = s[:i] + new_block.rstrip("\n") + "\n" + s[j_end:]
    paper_path.write_text(out)


def main() -> int:
    ap = argparse.ArgumentParser()
    ap.add_argument(
        "--results-dir",
        default="benchmarks/nextstat-public-benchmarks/suites/bayesian/results_v10",
        help="Directory containing checked-in results_v10 artifacts.",
    )
    ap.add_argument(
        "--paper",
        default="docs/papers/nuts-progressive-sampling.md",
        help="Paper Markdown file to update.",
    )
    ap.add_argument(
        "--assets-dir",
        default="docs/papers/assets/nuts-v10",
        help="Directory for generated SVG assets.",
    )
    args = ap.parse_args()

    results_dir = Path(args.results_dir).resolve()
    paper_path = Path(args.paper).resolve()
    assets_dir = Path(args.assets_dir).resolve()

    rows = _collect_rows(results_dir)
    if not rows:
        raise SystemExit(f"no rows found under {results_dir}")

    # Compute for figures
    case_ids = ["glm_logistic_regression", "hier_random_intercept_non_centered", "eight_schools_non_centered"]

    def mean_for(case: str, backend: str) -> float:
        vals = [r.min_ess_bulk_per_sec for r in rows if r.case_id == case and r.backend == backend and r.min_ess_bulk_per_sec is not None]
        if not vals:
            return 0.0
        return float(sum(vals) / len(vals))

    ns_vals = [mean_for(c, "nextstat") for c in case_ids]
    st_vals = [mean_for(c, "cmdstanpy") for c in case_ids]

    _render_bar_svg(out_path=assets_dir / "ess_sec_bar.svg", cases=case_ids, ns_vals=ns_vals, st_vals=st_vals)
    _render_seed_scatter_svg(out_path=assets_dir / "ess_sec_seed_scatter.svg", rows=rows, case_ids=case_ids)

    md = _render_autogen_markdown(rows=rows, results_dir=results_dir)
    _replace_autogen_block(paper_path=paper_path, new_block=md)
    return 0


if __name__ == "__main__":
    raise SystemExit(main())
