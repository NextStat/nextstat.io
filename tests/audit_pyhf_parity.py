#!/usr/bin/env python3
"""Audit NextStat vs pyhf on local JSON workspaces.

This is a CLI (not pytest) harness that:
- Loads candidate JSON files (default: tests/fixtures/*.json)
- Filters to pyhf-style workspaces (channels/observations/measurements/version)
- Compares NLL parity (pyhf twice_nll vs 2 * nextstat nll) on init + shifted params
- Optionally compares MLE fits (best-fit params + NLL), with name alignment
- Records timings (model build, NLL eval throughput, fit wall time)

Outputs:
- JSON report (machine-readable)
- Markdown summary (human-readable)
"""

from __future__ import annotations

import argparse
import dataclasses
import json
import math
import statistics
import time
import traceback
from pathlib import Path
from typing import Any, Iterable


def _is_pyhf_workspace(obj: Any) -> bool:
    if not isinstance(obj, dict):
        return False
    required = {"channels", "observations", "measurements", "version"}
    if not required.issubset(set(obj.keys())):
        return False
    if not isinstance(obj.get("channels"), list):
        return False
    if not isinstance(obj.get("observations"), list):
        return False
    if not isinstance(obj.get("measurements"), list) or not obj["measurements"]:
        return False
    return True


def _load_json(path: Path) -> Any:
    return json.loads(path.read_text())


def _measurement_name(ws: dict[str, Any]) -> str:
    ms = ws.get("measurements", [])
    if not ms:
        raise ValueError("workspace has no measurements")
    name = ms[0].get("name")
    if not isinstance(name, str) or not name:
        raise ValueError("workspace measurement has no name")
    return name


def _map_params_by_name(src_names: list[str], src_params: list[float], dst_names: list[str], dst_init: list[float]) -> list[float]:
    dst_index = {name: i for i, name in enumerate(dst_names)}
    out = list(dst_init)
    for name, value in zip(src_names, src_params):
        if name not in dst_index:
            raise KeyError(f"Destination missing parameter '{name}'")
        out[dst_index[name]] = float(value)
    return out


def _shift_params(params: list[float], bounds: list[tuple[float | None, float | None]], shift: float) -> list[float]:
    out: list[float] = []
    for x, (lo, hi) in zip(params, bounds):
        lo_f = float("-inf") if lo is None else float(lo)
        hi_f = float("inf") if hi is None else float(hi)
        y = float(x) + float(shift)
        if y < lo_f:
            y = lo_f
        if y > hi_f:
            y = hi_f
        out.append(y)
    return out


def _bench(fn, iters: int) -> dict[str, float]:
    if iters <= 0:
        return {"iters": 0, "mean_s": float("nan"), "p50_s": float("nan"), "p90_s": float("nan")}
    times: list[float] = []
    for _ in range(iters):
        t0 = time.perf_counter()
        _ = fn()
        times.append(time.perf_counter() - t0)
    times_sorted = sorted(times)
    p50 = times_sorted[len(times_sorted) // 2]
    p90 = times_sorted[int(math.floor(0.9 * (len(times_sorted) - 1)))]
    return {
        "iters": float(iters),
        "mean_s": float(statistics.mean(times)),
        "p50_s": float(p50),
        "p90_s": float(p90),
    }


@dataclasses.dataclass
class WorkspaceResult:
    path: str
    ok: bool
    reason: str | None
    meta: dict[str, Any]
    timings: dict[str, Any]
    parity: dict[str, Any]
    diffs: dict[str, Any]
    errors: list[dict[str, str]]


def _top_param_diffs(pyhf_names: list[str], pyhf_params: list[float], ns_names: list[str], ns_params: list[float], top_n: int) -> list[dict[str, float | str]]:
    ns_index = {n: i for i, n in enumerate(ns_names)}
    rows: list[tuple[float, str, float, float]] = []
    for n, pv in zip(pyhf_names, pyhf_params):
        if n not in ns_index:
            continue
        nv = float(ns_params[ns_index[n]])
        pv_f = float(pv)
        rows.append((abs(pv_f - nv), n, pv_f, nv))
    rows.sort(key=lambda r: -r[0])
    out: list[dict[str, float | str]] = []
    for d, n, pv, nv in rows[:top_n]:
        out.append({"name": n, "abs_diff": float(d), "pyhf": float(pv), "nextstat": float(nv)})
    return out


def audit_workspace(
    workspace_path: Path,
    *,
    nll_bench_iters: int,
    shift: float,
    top_n_diffs: int,
    do_fit: bool,
    fit_max_params: int | None,
    nextstat_mle: dict[str, Any],
    pyhf_fit_options: dict[str, Any],
) -> WorkspaceResult:
    errors: list[dict[str, str]] = []
    timings: dict[str, Any] = {}
    parity: dict[str, Any] = {}
    diffs: dict[str, Any] = {}

    try:
        t0 = time.perf_counter()
        ws = _load_json(workspace_path)
        timings["load_json_s"] = time.perf_counter() - t0
    except Exception:
        return WorkspaceResult(
            path=str(workspace_path),
            ok=False,
            reason="json_parse_error",
            meta={},
            timings={},
            parity={},
            diffs={},
            errors=[{"where": "load_json", "traceback": traceback.format_exc()}],
        )

    if not _is_pyhf_workspace(ws):
        return WorkspaceResult(
            path=str(workspace_path),
            ok=False,
            reason="not_pyhf_workspace",
            meta={"keys": sorted(list(ws.keys())) if isinstance(ws, dict) else str(type(ws))},
            timings=timings,
            parity={},
            diffs={},
            errors=[],
        )

    try:
        import pyhf  # type: ignore
        import nextstat  # type: ignore
    except Exception:
        return WorkspaceResult(
            path=str(workspace_path),
            ok=False,
            reason="missing_deps",
            meta={},
            timings=timings,
            parity={},
            diffs={},
            errors=[{"where": "imports", "traceback": traceback.format_exc()}],
        )

    measurement_name = _measurement_name(ws)
    meta = {
        "measurement_name": measurement_name,
        "n_channels": len(ws.get("channels", [])),
        "n_observations": len(ws.get("observations", [])),
        "n_measurements": len(ws.get("measurements", [])),
        "version": ws.get("version"),
    }

    # pyhf model
    try:
        t0 = time.perf_counter()
        ws_pyhf = pyhf.Workspace(ws)
        model_pyhf = ws_pyhf.model(
            measurement_name=measurement_name,
            modifier_settings={
                "normsys": {"interpcode": "code4"},
                "histosys": {"interpcode": "code4p"},
            },
        )
        data_pyhf = ws_pyhf.data(model_pyhf)
        timings["pyhf_model_s"] = time.perf_counter() - t0
    except Exception:
        errors.append({"where": "pyhf_model", "traceback": traceback.format_exc()})
        return WorkspaceResult(
            path=str(workspace_path),
            ok=False,
            reason="pyhf_model_error",
            meta=meta,
            timings=timings,
            parity={},
            diffs={},
            errors=errors,
        )

    # nextstat model
    try:
        t0 = time.perf_counter()
        model_ns = nextstat.HistFactoryModel.from_workspace(json.dumps(ws))
        timings["nextstat_model_s"] = time.perf_counter() - t0
    except Exception:
        errors.append({"where": "nextstat_model", "traceback": traceback.format_exc()})
        return WorkspaceResult(
            path=str(workspace_path),
            ok=False,
            reason="nextstat_model_error",
            meta=meta,
            timings=timings,
            parity={},
            diffs={},
            errors=errors,
        )

    pyhf_names = list(model_pyhf.config.par_names)
    pyhf_init = [float(x) for x in model_pyhf.config.suggested_init()]
    pyhf_bounds = list(model_pyhf.config.suggested_bounds())

    ns_names = list(model_ns.parameter_names())
    ns_init = [float(x) for x in model_ns.suggested_init()]

    meta["n_params"] = len(pyhf_names)

    diffs["param_set_equal"] = set(pyhf_names) == set(ns_names)
    if not diffs["param_set_equal"]:
        diffs["only_pyhf"] = sorted(list(set(pyhf_names) - set(ns_names)))[:50]
        diffs["only_nextstat"] = sorted(list(set(ns_names) - set(pyhf_names)))[:50]

    # NLL parity at init
    try:
        ns_params_init = _map_params_by_name(pyhf_names, pyhf_init, ns_names, ns_init)
    except Exception:
        errors.append({"where": "map_params_init", "traceback": traceback.format_exc()})
        return WorkspaceResult(
            path=str(workspace_path),
            ok=False,
            reason="param_mapping_error",
            meta=meta,
            timings=timings,
            parity={},
            diffs=diffs,
            errors=errors,
        )

    def pyhf_twice_nll_init() -> float:
        return float(pyhf.infer.mle.twice_nll(pyhf_init, data_pyhf, model_pyhf).item())

    def ns_twice_nll_init() -> float:
        return 2.0 * float(model_ns.nll(ns_params_init))

    t0 = time.perf_counter()
    pyhf_val_init = pyhf_twice_nll_init()
    timings["pyhf_twice_nll_init_s"] = time.perf_counter() - t0

    t0 = time.perf_counter()
    ns_val_init = ns_twice_nll_init()
    timings["nextstat_twice_nll_init_s"] = time.perf_counter() - t0

    parity["twice_nll_init"] = {"pyhf": pyhf_val_init, "nextstat": ns_val_init, "abs_diff": abs(pyhf_val_init - ns_val_init)}
    timings["bench_pyhf_twice_nll_init"] = _bench(pyhf_twice_nll_init, nll_bench_iters)
    timings["bench_nextstat_twice_nll_init"] = _bench(ns_twice_nll_init, nll_bench_iters)

    # NLL parity at shifted params
    pyhf_shift = _shift_params(pyhf_init, pyhf_bounds, shift=shift)
    try:
        ns_params_shift = _map_params_by_name(pyhf_names, pyhf_shift, ns_names, ns_init)
    except Exception:
        errors.append({"where": "map_params_shift", "traceback": traceback.format_exc()})
        return WorkspaceResult(
            path=str(workspace_path),
            ok=False,
            reason="param_mapping_error",
            meta=meta,
            timings=timings,
            parity=parity,
            diffs=diffs,
            errors=errors,
        )

    def pyhf_twice_nll_shift() -> float:
        return float(pyhf.infer.mle.twice_nll(pyhf_shift, data_pyhf, model_pyhf).item())

    def ns_twice_nll_shift() -> float:
        return 2.0 * float(model_ns.nll(ns_params_shift))

    t0 = time.perf_counter()
    pyhf_val_shift = pyhf_twice_nll_shift()
    timings["pyhf_twice_nll_shift_s"] = time.perf_counter() - t0

    t0 = time.perf_counter()
    ns_val_shift = ns_twice_nll_shift()
    timings["nextstat_twice_nll_shift_s"] = time.perf_counter() - t0

    parity["twice_nll_shift"] = {"pyhf": pyhf_val_shift, "nextstat": ns_val_shift, "abs_diff": abs(pyhf_val_shift - ns_val_shift)}

    # Optional: MLE fit parity (best-fit + NLL)
    if do_fit and (fit_max_params is None or len(pyhf_names) <= fit_max_params):
        try:
            t0 = time.perf_counter()
            pyhf_bestfit = [float(x) for x in pyhf.infer.mle.fit(data_pyhf, model_pyhf, **pyhf_fit_options)]
            timings["pyhf_fit_s"] = time.perf_counter() - t0
            pyhf_nll_hat = (
                float(pyhf.infer.mle.twice_nll(pyhf_bestfit, data_pyhf, model_pyhf).item()) / 2.0
            )
        except Exception:
            errors.append({"where": "pyhf_fit", "traceback": traceback.format_exc()})
            pyhf_bestfit = []
            pyhf_nll_hat = float("nan")

        try:
            t0 = time.perf_counter()
            mle = nextstat.MaximumLikelihoodEstimator(**nextstat_mle)
            ns_fit = mle.fit(model_ns)
            timings["nextstat_fit_s"] = time.perf_counter() - t0
            ns_nll_hat = float(ns_fit.nll)
            parity["nextstat_fit_status"] = {
                "converged": bool(ns_fit.converged),
                "n_iter": int(ns_fit.n_iter),
                "n_fev": int(ns_fit.n_fev),
                "n_gev": int(ns_fit.n_gev),
            }
        except Exception:
            errors.append({"where": "nextstat_fit", "traceback": traceback.format_exc()})
            ns_fit = None
            ns_nll_hat = float("nan")

        parity["nll_hat"] = {
            "pyhf": pyhf_nll_hat,
            "nextstat": ns_nll_hat,
            "abs_diff": abs(pyhf_nll_hat - ns_nll_hat) if math.isfinite(pyhf_nll_hat) and math.isfinite(ns_nll_hat) else float("nan"),
        }

        if ns_fit is not None and pyhf_bestfit:
            # Cross-evaluate each bestfit in the other implementation (name-aligned).
            try:
                ns_bestfit_in_pyhf = _map_params_by_name(
                    ns_names, list(ns_fit.parameters), pyhf_names, pyhf_init
                )
                pyhf_bestfit_in_ns = _map_params_by_name(pyhf_names, pyhf_bestfit, ns_names, ns_init)
                parity["cross_eval"] = {
                    "pyhf_nll_at_nextstat_hat": float(
                        pyhf.infer.mle.twice_nll(ns_bestfit_in_pyhf, data_pyhf, model_pyhf).item()
                    )
                    / 2.0,
                    "nextstat_nll_at_pyhf_hat": float(model_ns.nll(pyhf_bestfit_in_ns)),
                }
            except Exception:
                errors.append({"where": "cross_eval", "traceback": traceback.format_exc()})

            try:
                diffs["top_bestfit_param_diffs"] = _top_param_diffs(
                    pyhf_names,
                    pyhf_bestfit,
                    ns_names,
                    list(ns_fit.parameters),
                    top_n=top_n_diffs,
                )
                if diffs["top_bestfit_param_diffs"]:
                    diffs["bestfit_max_abs_diff"] = float(diffs["top_bestfit_param_diffs"][0]["abs_diff"])
            except Exception:
                errors.append({"where": "param_diffs", "traceback": traceback.format_exc()})

            # Heuristic regression guard: huge uncertainties are almost always numerical fallback.
            try:
                suspicious = []
                for name, unc in zip(ns_names, list(ns_fit.uncertainties)):
                    u = float(unc)
                    if not math.isfinite(u) or u <= 0.0 or u >= 1e5 or abs(u - 1e6) < 1e-6:
                        suspicious.append({"name": name, "uncertainty": u})
                        if len(suspicious) >= 50:
                            break
                diffs["suspicious_uncertainties"] = suspicious
            except Exception:
                errors.append({"where": "uncertainties", "traceback": traceback.format_exc()})
    elif do_fit and fit_max_params is not None and len(pyhf_names) > fit_max_params:
        diffs["fit_skipped"] = f"n_params={len(pyhf_names)} > fit_max_params={fit_max_params}"

    # Verdict (tolerances match our deterministic contract for NLL parity)
    # For fit NLL, allow larger drift for big models due to optimizer differences.
    tol_twice_nll_abs = 1e-8
    tol_twice_nll_rel = 1e-6

    def _ok(ref: float, val: float) -> bool:
        d = abs(ref - val)
        return d <= max(tol_twice_nll_abs, tol_twice_nll_rel * max(abs(ref), abs(val), 1.0))

    ok_init = _ok(pyhf_val_init, ns_val_init)
    ok_shift = _ok(pyhf_val_shift, ns_val_shift)
    ok = bool(ok_init and ok_shift)
    reason = None if ok else "nll_parity_failed"

    return WorkspaceResult(
        path=str(workspace_path),
        ok=ok,
        reason=reason,
        meta=meta,
        timings=timings,
        parity=parity,
        diffs=diffs,
        errors=errors,
    )


def _iter_default_workspace_paths() -> Iterable[Path]:
    fixtures = Path("tests/fixtures")
    if fixtures.exists():
        for p in sorted(fixtures.glob("*.json")):
            yield p


def _write_json(path: Path, data: Any) -> None:
    path.parent.mkdir(parents=True, exist_ok=True)
    path.write_text(json.dumps(data, indent=2, sort_keys=True) + "\n")


def _write_md(path: Path, report: dict[str, Any]) -> None:
    path.parent.mkdir(parents=True, exist_ok=True)

    rows = report["workspaces"]
    lines: list[str] = []
    lines.append("# pyhf parity audit (NextStat vs pyhf)")
    lines.append("")
    lines.append(f"Generated: `{report.get('generated_at')}`")
    lines.append("")
    lines.append("## Summary")
    lines.append("")
    lines.append(f"- Total: {report['summary']['n_total']}")
    lines.append(f"- pyhf workspaces: {report['summary']['n_pyhf_workspaces']}")
    lines.append(f"- Passed NLL parity: {report['summary']['n_ok']}")
    lines.append(f"- Failed/other: {report['summary']['n_bad']}")
    lines.append("")
    lines.append("## Results")
    lines.append("")
    lines.append("| Workspace | Status | n_params | Init Δtwice_nll | Shift Δtwice_nll | Δnll_hat | Max |Δ bestfit| | pyhf fit (s) | NextStat fit (s) | Notes |")
    lines.append("|---|---:|---:|---:|---:|---:|---:|---:|---:|---|")
    for r in rows:
        rel = r["path"]
        status = "OK" if r["ok"] else (r.get("reason") or "FAIL")
        n_params = r.get("meta", {}).get("n_params", float("nan"))
        init = r.get("parity", {}).get("twice_nll_init", {}).get("abs_diff", float("nan"))
        sh = r.get("parity", {}).get("twice_nll_shift", {}).get("abs_diff", float("nan"))
        dnll_hat = r.get("parity", {}).get("nll_hat", {}).get("abs_diff", float("nan"))
        max_d = r.get("diffs", {}).get("bestfit_max_abs_diff", float("nan"))
        t_pyhf = r.get("timings", {}).get("pyhf_fit_s", float("nan"))
        t_ns = r.get("timings", {}).get("nextstat_fit_s", float("nan"))
        notes = ""
        if r.get("reason") == "not_pyhf_workspace":
            notes = "skipped (not pyhf JSON)"
        if r.get("reason") == "json_parse_error":
            notes = "invalid JSON"
        if r.get("diffs", {}).get("param_set_equal") is False:
            notes = (notes + "; " if notes else "") + "param set mismatch"
        fit_status = r.get("parity", {}).get("nextstat_fit_status")
        if isinstance(fit_status, dict) and fit_status.get("converged") is False:
            n_iter = fit_status.get("n_iter")
            extra = f"nextstat not converged (n_iter={n_iter})" if isinstance(n_iter, int) else "nextstat not converged"
            notes = (notes + "; " if notes else "") + extra
        nll_hat = r.get("parity", {}).get("nll_hat")
        if isinstance(nll_hat, dict) and isinstance(nll_hat.get("pyhf"), (int, float)) and isinstance(
            nll_hat.get("nextstat"), (int, float)
        ):
            if float(nll_hat["nextstat"]) < float(nll_hat["pyhf"]):
                notes = (notes + "; " if notes else "") + "nextstat nll < pyhf"
            elif float(nll_hat["nextstat"]) > float(nll_hat["pyhf"]):
                notes = (notes + "; " if notes else "") + "pyhf nll < nextstat"
        lines.append(
            f"| `{rel}` | {status} | {n_params} | {init:.3e} | {sh:.3e} | {dnll_hat:.3e} | {max_d:.3e} | {t_pyhf:.2f} | {t_ns:.2f} | {notes} |"
        )

    bestfit_bad = [
        r
        for r in rows
        if r.get("ok")
        and isinstance(r.get("diffs", {}).get("bestfit_max_abs_diff"), (int, float))
        and r["diffs"]["bestfit_max_abs_diff"] > 1e-2
    ]
    if bestfit_bad:
        lines.append("")
        lines.append("## Best-fit mismatches (details)")
        lines.append("")
        bestfit_bad = sorted(bestfit_bad, key=lambda r: -float(r["diffs"]["bestfit_max_abs_diff"]))
        for r in bestfit_bad[:10]:
            lines.append(f"### `{r['path']}`")
            lines.append("")
            lines.append(f"- Init Δtwice_nll: {r['parity']['twice_nll_init']['abs_diff']:.6e}")
            lines.append(f"- Shift Δtwice_nll: {r['parity']['twice_nll_shift']['abs_diff']:.6e}")
            if "nll_hat" in r.get("parity", {}):
                nh = r["parity"]["nll_hat"]
                lines.append(
                    f"- nll_hat: pyhf={nh.get('pyhf', float('nan')):.6g}, nextstat={nh.get('nextstat', float('nan')):.6g} (|Δ|={nh.get('abs_diff', float('nan')):.6e})"
                )
            if "nextstat_fit_status" in r.get("parity", {}):
                fs = r["parity"]["nextstat_fit_status"]
                if isinstance(fs, dict):
                    lines.append(
                        "- NextStat fit status: "
                        + ", ".join(
                            [
                                f"converged={fs.get('converged')}",
                                f"n_iter={fs.get('n_iter')}",
                                f"n_fev={fs.get('n_fev')}",
                                f"n_gev={fs.get('n_gev')}",
                            ]
                        )
                    )
            if "cross_eval" in r.get("parity", {}):
                ce = r["parity"]["cross_eval"]
                if isinstance(ce, dict):
                    lines.append(
                        "- Cross-eval: "
                        + ", ".join(
                            [
                                f"pyhf_nll(nextstat_hat)={ce.get('pyhf_nll_at_nextstat_hat', float('nan')):.6g}",
                                f"nextstat_nll(pyhf_hat)={ce.get('nextstat_nll_at_pyhf_hat', float('nan')):.6g}",
                            ]
                        )
                    )
            if "top_bestfit_param_diffs" in r.get("diffs", {}):
                lines.append("- Worst bestfit param diffs (top 10):")
                for d in r["diffs"]["top_bestfit_param_diffs"][:10]:
                    lines.append(
                        f"  - `{d['name']}`: |Δ|={d['abs_diff']:.3e} (pyhf={d['pyhf']:+.6g}, nextstat={d['nextstat']:+.6g})"
                    )
            lines.append("")

    path.write_text("\n".join(lines) + "\n")


def main(argv: list[str] | None = None) -> int:
    ap = argparse.ArgumentParser()
    ap.add_argument("--workspace", action="append", default=[], help="Workspace JSON path (repeatable). Default: tests/fixtures/*.json")
    ap.add_argument("--out-json", default="tmp/pyhf_parity_audit.json")
    ap.add_argument("--out-md", default="tmp/pyhf_parity_audit.md")
    ap.add_argument("--nll-bench-iters", type=int, default=25)
    ap.add_argument("--shift", type=float, default=0.123)
    ap.add_argument("--top-n-diffs", type=int, default=20)
    ap.add_argument("--fit", action="store_true", help="Also run MLE fits (can be slow).")
    ap.add_argument("--fit-max-params", type=int, default=600, help="Skip fits if n_params exceeds this threshold (use 0 to disable).")
    ap.add_argument("--nextstat-max-iter", type=int, default=1000)
    ap.add_argument("--nextstat-tol", type=float, default=1e-6)
    ap.add_argument("--nextstat-m", type=int, default=10)
    ap.add_argument("--pyhf-method", default="SLSQP")
    ap.add_argument("--pyhf-maxiter", type=int, default=100000)
    ap.add_argument("--pyhf-tolerance", type=float, default=float("nan"))
    ap.add_argument("--pyhf-do-grad", type=int, default=-1, help="Set to 0/1 to override pyhf do_grad, or -1 to keep default.")
    ap.add_argument("--pyhf-do-stitch", type=int, default=0, help="Set to 1 to enable pyhf do_stitch (helps some SciPy methods like L-BFGS-B).")
    args = ap.parse_args(argv)

    nextstat_mle = {"max_iter": int(args.nextstat_max_iter), "tol": float(args.nextstat_tol), "m": int(args.nextstat_m)}
    pyhf_fit_options: dict[str, Any] = {"method": str(args.pyhf_method), "maxiter": int(args.pyhf_maxiter)}
    if math.isfinite(float(args.pyhf_tolerance)):
        pyhf_fit_options["tolerance"] = float(args.pyhf_tolerance)
    if args.pyhf_do_grad in (0, 1):
        pyhf_fit_options["do_grad"] = bool(args.pyhf_do_grad)
    if args.pyhf_do_stitch in (0, 1):
        pyhf_fit_options["do_stitch"] = bool(args.pyhf_do_stitch)

    paths = [Path(p) for p in args.workspace] if args.workspace else list(_iter_default_workspace_paths())

    results: list[dict[str, Any]] = []
    n_total = 0
    n_pyhf = 0
    n_ok = 0

    for p in paths:
        n_total += 1
        print(f"[audit] {p}", flush=True)
        r = audit_workspace(
            p,
            nll_bench_iters=args.nll_bench_iters,
            shift=args.shift,
            top_n_diffs=args.top_n_diffs,
            do_fit=bool(args.fit),
            fit_max_params=None if args.fit_max_params == 0 else int(args.fit_max_params),
            nextstat_mle=nextstat_mle,
            pyhf_fit_options=pyhf_fit_options,
        )
        rd = dataclasses.asdict(r)
        results.append(rd)
        if rd.get("reason") not in ("json_parse_error", "not_pyhf_workspace"):
            n_pyhf += 1
        if rd.get("ok"):
            n_ok += 1

    report = {
        "generated_at": time.strftime("%Y-%m-%d %H:%M:%S"),
        "nextstat_mle_config": nextstat_mle,
        "pyhf_fit_options": pyhf_fit_options,
        "summary": {
            "n_total": n_total,
            "n_pyhf_workspaces": n_pyhf,
            "n_ok": n_ok,
            "n_bad": n_total - n_ok,
        },
        "workspaces": results,
    }

    _write_json(Path(args.out_json), report)
    _write_md(Path(args.out_md), report)

    print(f"Wrote: {args.out_json}")
    print(f"Wrote: {args.out_md}")

    # Non-zero exit if any parity failures on real pyhf workspaces.
    n_failed = sum(1 for r in results if r.get("reason") == "nll_parity_failed")
    return 1 if n_failed else 0


if __name__ == "__main__":
    raise SystemExit(main())
