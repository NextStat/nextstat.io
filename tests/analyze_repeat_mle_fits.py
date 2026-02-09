#!/usr/bin/env python3
"""Analyze `tests/repeat_mle_fits.py` JSONL logs into a publishable audit report.

This script is intentionally pure-standard-library so it can be run anywhere:

  PYTHONPATH=bindings/ns-py/python ./.venv/bin/python tests/analyze_repeat_mle_fits.py \
    --in-jsonl tmp/repeat_mle_fits.jsonl \
    --out-json tmp/repeat_mle_fits_audit.json \
    --out-md tmp/repeat_mle_fits_audit.md

Key ideas:
- We validate NextStat as a tool against pyhf as the trusted reference.
- For "stability" aggregates (mean/stdev of params/NLL), we only use paired runs
  where both tools succeeded and NextStat reported `converged=True`.
- We also report "all-ok" timing stats for transparency.
- Cross-eval checks compare objective values at each other's best-fit points.
"""

from __future__ import annotations

import argparse
import dataclasses
import datetime as dt
import json
import math
import statistics
from collections import defaultdict
from pathlib import Path
from typing import Any, Dict, Iterable, List, Optional, Tuple


def _percentiles(xs: List[float], ps: List[int]) -> Dict[str, float]:
    if not xs:
        return {f"p{p}": float("nan") for p in ps}
    xs_sorted = sorted(xs)
    out: Dict[str, float] = {}
    for p in ps:
        if p <= 0:
            out[f"p{p}"] = float(xs_sorted[0])
            continue
        if p >= 100:
            out[f"p{p}"] = float(xs_sorted[-1])
            continue
        k = int(math.floor((p / 100.0) * (len(xs_sorted) - 1)))
        out[f"p{p}"] = float(xs_sorted[k])
    return out


def _tool_stats(rows: List[dict[str, Any]], tool_key: str) -> dict[str, Any]:
    wall = [float(r[tool_key]["fit_wall_s"]) for r in rows]
    cpu = [float(r[tool_key]["fit_cpu_s"]) for r in rows]
    nll = [float(r[tool_key]["nll"]) for r in rows]
    out: dict[str, Any] = {
        "n": int(len(rows)),
        "fit_wall_s": {
            "mean": float(statistics.mean(wall)) if wall else float("nan"),
            "stdev": float(statistics.pstdev(wall)) if len(wall) >= 2 else 0.0,
            **_percentiles(wall, [50, 90, 99]),
        },
        "fit_cpu_s": {
            "mean": float(statistics.mean(cpu)) if cpu else float("nan"),
            "stdev": float(statistics.pstdev(cpu)) if len(cpu) >= 2 else 0.0,
            **_percentiles(cpu, [50, 90, 99]),
        },
        "nll": {
            "mean": float(statistics.mean(nll)) if nll else float("nan"),
            "stdev": float(statistics.pstdev(nll)) if len(nll) >= 2 else 0.0,
            "min": float(min(nll)) if nll else float("nan"),
            "max": float(max(nll)) if nll else float("nan"),
        },
    }
    if tool_key == "nextstat":
        conv = [bool(r[tool_key].get("converged")) for r in rows if r[tool_key].get("converged") is not None]
        n_conv = sum(1 for c in conv if c)
        out["n_converged"] = int(n_conv)
        out["converged_rate"] = float(n_conv / len(conv)) if conv else float("nan")
        # Optional iteration counters if present.
        iters = [int(r[tool_key]["n_iter"]) for r in rows if r[tool_key].get("n_iter") is not None]
        if iters:
            out["n_iter"] = {
                "mean": float(statistics.mean(iters)),
                "stdev": float(statistics.pstdev(iters)) if len(iters) >= 2 else 0.0,
                **_percentiles([float(x) for x in iters], [50, 90, 99]),
            }
    return out


def _mean_params(rows: List[dict[str, Any]], tool_key: str, names: List[str]) -> List[float]:
    n = len(names)
    if n == 0:
        return []
    sums = [0.0] * n
    used = 0
    for r in rows:
        ps = r[tool_key].get("params")
        if not isinstance(ps, list) or len(ps) != n:
            continue
        used += 1
        for i, v in enumerate(ps):
            sums[i] += float(v)
    if used == 0:
        return [float("nan")] * n
    return [s / used for s in sums]


def _top_abs_diffs(
    pyhf_names: List[str],
    pyhf_mean: List[float],
    ns_names: List[str],
    ns_mean: List[float],
    top_n: int,
) -> List[dict[str, float | str]]:
    ns_index = {n: i for i, n in enumerate(ns_names)}
    rows: List[Tuple[float, str, float, float]] = []
    for name, pv in zip(pyhf_names, pyhf_mean):
        if name not in ns_index:
            continue
        nv = float(ns_mean[ns_index[name]])
        pv_f = float(pv)
        rows.append((abs(pv_f - nv), name, pv_f, nv))
    rows.sort(reverse=True, key=lambda t: t[0])
    out: List[dict[str, float | str]] = []
    for d, name, pv, nv in rows[: max(0, int(top_n))]:
        out.append({"name": name, "abs_diff": float(d), "pyhf": float(pv), "nextstat": float(nv)})
    return out


def _cross_eval(rows: List[dict[str, Any]]) -> dict[str, Any]:
    # Deltas should be ~0 if both tool NLL implementations match.
    d_pyhf_vs_ns_at_ns_hat: List[float] = []
    d_ns_vs_pyhf_at_pyhf_hat: List[float] = []

    for r in rows:
        cross = r.get("cross_eval") or {}
        if "error" in cross:
            continue
        pyhf_at_ns = cross.get("pyhf_nll_at_nextstat_hat")
        ns_at_pyhf = cross.get("nextstat_nll_at_pyhf_hat")
        if pyhf_at_ns is not None and math.isfinite(float(pyhf_at_ns)) and math.isfinite(float(r["nextstat"]["nll"])):
            d_pyhf_vs_ns_at_ns_hat.append(float(pyhf_at_ns) - float(r["nextstat"]["nll"]))
        if ns_at_pyhf is not None and math.isfinite(float(ns_at_pyhf)) and math.isfinite(float(r["pyhf"]["nll"])):
            d_ns_vs_pyhf_at_pyhf_hat.append(float(ns_at_pyhf) - float(r["pyhf"]["nll"]))

    def stats(xs: List[float]) -> dict[str, Any]:
        if not xs:
            return {"n": 0, "mean": float("nan"), "stdev": float("nan"), "max_abs": float("nan")}
        return {
            "n": int(len(xs)),
            "mean": float(statistics.mean(xs)),
            "stdev": float(statistics.pstdev(xs)) if len(xs) >= 2 else 0.0,
            "max_abs": float(max(abs(x) for x in xs)),
        }

    return {
        "pyhf_nll_at_nextstat_hat_minus_nextstat_nll": stats(d_pyhf_vs_ns_at_ns_hat),
        "nextstat_nll_at_pyhf_hat_minus_pyhf_nll": stats(d_ns_vs_pyhf_at_pyhf_hat),
    }


@dataclasses.dataclass
class WorkspaceLog:
    workspace_id: str
    path: str
    measurement: str
    pyhf_names: List[str]
    nextstat_names: List[str]
    runs: List[dict[str, Any]]
    errors: List[dict[str, Any]]


def _load_jsonl(path: Path) -> Tuple[dict[str, Any], dict[str, WorkspaceLog], list[dict[str, Any]], list[dict[str, Any]]]:
    header: dict[str, Any] = {}
    metas: dict[str, dict[str, Any]] = {}
    runs_by_ws: dict[str, list[dict[str, Any]]] = defaultdict(list)
    errors_by_ws: dict[str, list[dict[str, Any]]] = defaultdict(list)
    skipped: list[dict[str, Any]] = []
    workspace_errors: list[dict[str, Any]] = []

    with path.open("r", encoding="utf-8") as fp:
        for line in fp:
            rec = json.loads(line)
            t = rec.get("type")
            if t == "header":
                header = rec
            elif t == "workspace_meta":
                metas[str(rec["workspace_id"])] = rec
            elif t == "run":
                if bool(rec.get("warmup")):
                    continue
                runs_by_ws[str(rec["workspace_id"])].append(rec)
            elif t == "skipped":
                skipped.append(rec)
            elif t == "workspace_error":
                workspace_errors.append(rec)
                ws_id = rec.get("workspace_id")
                if ws_id is not None:
                    errors_by_ws[str(ws_id)].append(rec)

    workspaces: dict[str, WorkspaceLog] = {}
    for ws_id, meta in metas.items():
        workspaces[ws_id] = WorkspaceLog(
            workspace_id=ws_id,
            path=str(meta.get("path", "")),
            measurement=str(meta.get("measurement", "")),
            pyhf_names=list((meta.get("pyhf") or {}).get("names") or []),
            nextstat_names=list((meta.get("nextstat") or {}).get("names") or []),
            runs=runs_by_ws.get(ws_id, []),
            errors=errors_by_ws.get(ws_id, []),
        )

    return header, workspaces, skipped, workspace_errors


def _merge_headers(base: dict[str, Any], other: dict[str, Any]) -> None:
    # Enforce that the critical knobs match across input shards, otherwise the
    # merged statistics are meaningless/publishability suffers.
    keys = [
        "pyhf_version",
        "pyhf_backend",
        "pyhf_optimizer",
        "nextstat_version",
        "threads",
        "toy_main",
        "toy_seed",
        "fit_order",
        "pyhf_fit_options",
        "nextstat_mle_config",
    ]
    for k in keys:
        if k in base and k in other and base[k] != other[k]:
            raise SystemExit(f"Cannot merge JSONL logs: header mismatch for '{k}': {base[k]!r} != {other[k]!r}")


def _load_many_jsonl(paths: List[Path]) -> Tuple[dict[str, Any], dict[str, WorkspaceLog], list[dict[str, Any]], list[dict[str, Any]]]:
    header: dict[str, Any] = {}
    metas: dict[str, dict[str, Any]] = {}
    runs_by_ws: dict[str, list[dict[str, Any]]] = defaultdict(list)
    errors_by_ws: dict[str, list[dict[str, Any]]] = defaultdict(list)
    skipped: list[dict[str, Any]] = []
    workspace_errors: list[dict[str, Any]] = []

    for p in paths:
        h, ws_map, sk, errs = _load_jsonl(p)
        if not header:
            header = h
        elif h:
            _merge_headers(header, h)

        # Reconstruct the raw pieces from the parsed structure so we can merge
        # without losing runs for workspaces that appear in multiple shards.
        for ws_id, ws in ws_map.items():
            if ws_id not in metas:
                metas[ws_id] = {
                    "workspace_id": ws.workspace_id,
                    "path": ws.path,
                    "measurement": ws.measurement,
                    "pyhf": {"names": ws.pyhf_names},
                    "nextstat": {"names": ws.nextstat_names},
                }
            runs_by_ws[ws_id].extend(ws.runs)
            errors_by_ws[ws_id].extend(ws.errors)

        skipped.extend(sk)
        workspace_errors.extend(errs)

    workspaces: dict[str, WorkspaceLog] = {}
    for ws_id, meta in metas.items():
        workspaces[ws_id] = WorkspaceLog(
            workspace_id=ws_id,
            path=str(meta.get("path", "")),
            measurement=str(meta.get("measurement", "")),
            pyhf_names=list((meta.get("pyhf") or {}).get("names") or []),
            nextstat_names=list((meta.get("nextstat") or {}).get("names") or []),
            runs=runs_by_ws.get(ws_id, []),
            errors=errors_by_ws.get(ws_id, []),
        )

    return header, workspaces, skipped, workspace_errors


def _summarize_workspace(ws: WorkspaceLog) -> dict[str, Any]:
    rows = ws.runs
    pyhf_ok = [r for r in rows if r.get("pyhf", {}).get("ok")]
    ns_ok = [r for r in rows if r.get("nextstat", {}).get("ok")]
    paired_ok = [r for r in rows if r.get("pyhf", {}).get("ok") and r.get("nextstat", {}).get("ok")]
    paired_converged = [r for r in paired_ok if r.get("nextstat", {}).get("converged") is True]

    py_sel = _tool_stats(paired_converged, "pyhf")
    ns_sel = _tool_stats(paired_converged, "nextstat")
    py_all = _tool_stats(pyhf_ok, "pyhf")
    ns_all = _tool_stats(ns_ok, "nextstat")

    py_mean = _mean_params(paired_converged, "pyhf", ws.pyhf_names)
    ns_mean = _mean_params(paired_converged, "nextstat", ws.nextstat_names)

    max_abs_diffs = [
        float(r.get("diffs", {}).get("bestfit_max_abs_diff"))
        for r in paired_ok
        if r.get("diffs", {}).get("bestfit_max_abs_diff") is not None
    ]

    cross_sel = _cross_eval(paired_converged)
    cross_all_ok = _cross_eval(paired_ok)

    return {
        "path": ws.path,
        "measurement": ws.measurement,
        "selection": {
            "n_total_runs": int(len(rows)),
            "n_pyhf_ok": int(len(pyhf_ok)),
            "n_nextstat_ok": int(len(ns_ok)),
            "n_paired_ok": int(len(paired_ok)),
            "n_nextstat_converged": int(len(paired_converged)),
        },
        "selected": {"pyhf": py_sel, "nextstat": ns_sel},
        "all_ok": {"pyhf": py_all, "nextstat": ns_all},
        "cross_eval": {"selected": cross_sel, "all_ok": cross_all_ok},
        "params": {
            "pyhf_names": ws.pyhf_names,
            "nextstat_names": ws.nextstat_names,
            "top_abs_mean_diffs_pyhf_order": _top_abs_diffs(ws.pyhf_names, py_mean, ws.nextstat_names, ns_mean, top_n=20),
            "bestfit_max_abs_diff_all_ok": {
                "n": int(len(max_abs_diffs)),
                "mean": float(statistics.mean(max_abs_diffs)) if max_abs_diffs else float("nan"),
                "stdev": float(statistics.pstdev(max_abs_diffs)) if len(max_abs_diffs) >= 2 else 0.0,
                **_percentiles(max_abs_diffs, [50, 90, 99]),
                "max": float(max(max_abs_diffs)) if max_abs_diffs else float("nan"),
            },
        },
        "errors": ws.errors,
    }


def _write_md(out_md: Path, *, header: dict[str, Any], summaries: list[dict[str, Any]], skipped: list[dict[str, Any]], workspace_errors: list[dict[str, Any]]) -> None:
    lines: list[str] = []
    lines.append("# Repeated MLE fits audit (pyhf vs NextStat)")
    lines.append("")
    lines.append(f"Generated: `{dt.datetime.now().strftime('%Y-%m-%d %H:%M:%S')}`")
    lines.append("")

    if header:
        lines.append("## Run config")
        lines.append("")
        for k in [
            "n_runs",
            "n_warmup",
            "threads",
            "excluded",
            "pyhf_version",
            "pyhf_backend",
            "pyhf_optimizer",
            "nextstat_version",
            "pyhf_fit_options",
            "nextstat_mle_config",
        ]:
            if k in header:
                lines.append(f"- {k}: `{header[k]}`")
        lines.append("")

    lines.append("## Summary")
    lines.append("")
    lines.append("| Workspace | selected (paired+converged) | pyhf mean/p50/p90 (s) | ns mean/p50/p90 (s) | speedup | ΔNLL (ns-pyhf) | max|Δobj| @ optima |")
    lines.append("|---|---:|---:|---:|---:|---:|---:|")

    for s in summaries:
        sel = s["selection"]
        py = s["selected"]["pyhf"]
        ns = s["selected"]["nextstat"]
        py_wall = py["fit_wall_s"]
        ns_wall = ns["fit_wall_s"]

        n_sel = int(sel["n_nextstat_converged"])
        n_tot = int(sel["n_total_runs"])
        py_t = float(py_wall["mean"])
        ns_t = float(ns_wall["mean"])
        speed = (py_t / ns_t) if (math.isfinite(py_t) and math.isfinite(ns_t) and ns_t > 0) else float("nan")
        d_nll = float(ns["nll"]["mean"]) - float(py["nll"]["mean"])

        obj1 = s["cross_eval"]["selected"]["pyhf_nll_at_nextstat_hat_minus_nextstat_nll"]["max_abs"]
        obj2 = s["cross_eval"]["selected"]["nextstat_nll_at_pyhf_hat_minus_pyhf_nll"]["max_abs"]
        max_obj = float(max(obj1, obj2)) if (math.isfinite(float(obj1)) and math.isfinite(float(obj2))) else float("nan")

        lines.append(
            f"| `{s['path']}` | {n_sel}/{n_tot} | "
            f"{py_t:.6f}/{float(py_wall['p50']):.6f}/{float(py_wall['p90']):.6f} | "
            f"{ns_t:.6f}/{float(ns_wall['p50']):.6f}/{float(ns_wall['p90']):.6f} | "
            f"{speed:.2f}x | {d_nll:.6g} | {max_obj:.3g} |"
        )

    lines.append("")

    if skipped:
        lines.append("## Skipped inputs")
        lines.append("")
        for rec in skipped:
            lines.append(f"- `{rec.get('path')}`: {rec.get('reason')}")
        lines.append("")

    if workspace_errors:
        lines.append("## Workspace errors")
        lines.append("")
        for rec in workspace_errors[:20]:
            where = rec.get("where")
            path = rec.get("path")
            lines.append(f"- `{path}` ({where})")
        if len(workspace_errors) > 20:
            lines.append(f"- ... ({len(workspace_errors) - 20} more)")
        lines.append("")

    lines.append("## Per-workspace details")
    for s in summaries:
        lines.append("")
        lines.append(f"### `{s['path']}`")
        lines.append("")
        lines.append(f"- measurement: `{s.get('measurement')}`")
        lines.append(f"- selection: `{s['selection']}`")
        lines.append(f"- cross-eval (selected): `{s['cross_eval']['selected']}`")
        lines.append("")

        top = s["params"]["top_abs_mean_diffs_pyhf_order"]
        if top:
            lines.append("Top |Δmean| params (pyhf name order):")
            for row in top[:20]:
                lines.append(f"- `{row['name']}`: |Δmean|={float(row['abs_diff']):.6g} (pyhf={float(row['pyhf']):.6g}, ns={float(row['nextstat']):.6g})")
            lines.append("")

        lines.append(f"Per-run bestfit max|Δ| (all paired-ok runs): `{s['params']['bestfit_max_abs_diff_all_ok']}`")
        lines.append("")

    out_md.parent.mkdir(parents=True, exist_ok=True)
    out_md.write_text("\n".join(lines) + "\n", encoding="utf-8")


def main(argv: Optional[List[str]] = None) -> int:
    ap = argparse.ArgumentParser()
    ap.add_argument("--in-jsonl", type=Path, action="append", required=True, help="Input JSONL log (repeatable, to merge shards).")
    ap.add_argument("--out-json", type=Path, default=Path("tmp/repeat_mle_fits_audit.json"))
    ap.add_argument("--out-md", type=Path, default=Path("tmp/repeat_mle_fits_audit.md"))
    args = ap.parse_args(argv)

    header, workspaces, skipped, workspace_errors = _load_many_jsonl(list(args.in_jsonl))
    summaries: list[dict[str, Any]] = []
    for ws in sorted(workspaces.values(), key=lambda w: w.path):
        if not ws.runs:
            continue
        summaries.append(_summarize_workspace(ws))

    out = {"header": header, "workspaces": summaries, "skipped": skipped, "workspace_errors": workspace_errors}
    args.out_json.parent.mkdir(parents=True, exist_ok=True)
    args.out_json.write_text(json.dumps(out, indent=2, sort_keys=True) + "\n", encoding="utf-8")

    _write_md(args.out_md, header=header, summaries=summaries, skipped=skipped, workspace_errors=workspace_errors)
    print(f"Wrote: {args.out_json}")
    print(f"Wrote: {args.out_md}")
    return 0


if __name__ == "__main__":
    raise SystemExit(main())
