#!/usr/bin/env python3
"""Repeat MLE fits for pyhf-style JSON workspaces and log stability + speed.

This is a research-grade runner intended to answer:
- Are pyhf and NextStat deterministic across repeated fits on the same workspace?
- If not, what is the run-to-run variance (params + NLL)?
- How do fit wall-times compare (distribution, not just a single run)?

It writes a raw JSONL log (one record per run per workspace) and a summary JSON/MD
with means/stddevs and name-aligned parameter diffs.
"""

from __future__ import annotations

import argparse
import dataclasses
import hashlib
import json
import math
import os
import platform
import statistics
import subprocess
import sys
import time
import traceback
from pathlib import Path
from typing import Any, Dict, Iterable, List, Optional, Tuple

import numpy as np


def _git_meta() -> dict[str, Any]:
    try:
        head = (
            subprocess.run(
                ["git", "rev-parse", "HEAD"],
                check=True,
                capture_output=True,
                text=True,
            )
            .stdout.strip()
        )
    except Exception:
        head = None

    try:
        # Avoid dumping the whole diff; just record that the working tree is dirty
        # and how many paths are modified/untracked.
        status = subprocess.run(
            ["git", "status", "--porcelain"],
            check=True,
            capture_output=True,
            text=True,
        ).stdout
        dirty_paths = [ln for ln in status.splitlines() if ln.strip()]
        dirty = len(dirty_paths) > 0
        dirty_count = len(dirty_paths)
    except Exception:
        dirty = None
        dirty_count = None

    try:
        branch = (
            subprocess.run(
                ["git", "rev-parse", "--abbrev-ref", "HEAD"],
                check=True,
                capture_output=True,
                text=True,
            )
            .stdout.strip()
        )
    except Exception:
        branch = None

    return {"git_head": head, "git_branch": branch, "git_dirty": dirty, "git_dirty_count": dirty_count}


def _load_jsonl_records(path: Path) -> List[dict[str, Any]]:
    if not path.exists():
        return []
    out: List[dict[str, Any]] = []
    with path.open("r", encoding="utf-8") as fp:
        for line in fp:
            if not line.strip():
                continue
            try:
                out.append(json.loads(line))
            except Exception:
                # Ignore partial/corrupt trailing lines.
                continue
    return out


def _next_run_idx(existing: List[dict[str, Any]], ws_id: str) -> int:
    max_idx = -1
    for r in existing:
        if r.get("type") != "run":
            continue
        if str(r.get("workspace_id")) != ws_id:
            continue
        try:
            max_idx = max(max_idx, int(r.get("run_idx")))
        except Exception:
            continue
    return max_idx + 1


def _toy_seed_for_run(ws_id: str, toy_seed: int, run_idx: int) -> int:
    # Deterministic per-workspace-per-run seed so runs are resumable without
    # depending on RNG state progression.
    msg = f"{ws_id}::{toy_seed}::{run_idx}".encode("utf-8")
    h = hashlib.sha256(msg).digest()
    return int.from_bytes(h[:8], "little") & 0x7FFF_FFFF_FFFF_FFFF


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


def _measurement_name(ws: dict[str, Any]) -> str:
    ms = ws.get("measurements", [])
    if not ms:
        raise ValueError("workspace has no measurements")
    name = ms[0].get("name")
    if not isinstance(name, str) or not name:
        raise ValueError("workspace measurement has no name")
    return name


def _load_json(path: Path) -> Any:
    return json.loads(path.read_text())


def _iter_default_workspace_paths() -> Iterable[Path]:
    fixtures = Path("tests/fixtures")
    if fixtures.exists():
        for p in sorted(fixtures.glob("*.json")):
            yield p


def _map_params_by_name(
    src_names: List[str],
    src_params: List[float],
    dst_names: List[str],
    dst_init: List[float],
) -> List[float]:
    dst_index = {name: i for i, name in enumerate(dst_names)}
    out = list(dst_init)
    for name, value in zip(src_names, src_params):
        if name not in dst_index:
            raise KeyError(f"Destination missing parameter '{name}'")
        out[dst_index[name]] = float(value)
    return out


def _remap_params(src_names: List[str], src_params: List[float], dst_names: List[str]) -> List[float]:
    dst_index = {name: i for i, name in enumerate(dst_names)}
    out = [0.0] * len(dst_names)
    for name, value in zip(src_names, src_params):
        if name not in dst_index:
            raise KeyError(f"Destination missing parameter '{name}'")
        out[dst_index[name]] = float(value)
    if len(out) != len(dst_names):
        raise AssertionError("internal remap error")
    return out


def _max_abs_diff_by_name(
    pyhf_names: List[str],
    pyhf_params: List[float],
    ns_names: List[str],
    ns_params: List[float],
) -> Tuple[float, Optional[str]]:
    ns_index = {n: i for i, n in enumerate(ns_names)}
    worst = 0.0
    worst_name: Optional[str] = None
    for n, pv in zip(pyhf_names, pyhf_params):
        i = ns_index.get(n)
        if i is None:
            continue
        dv = abs(float(pv) - float(ns_params[i]))
        if dv > worst:
            worst = dv
            worst_name = n
    return (float(worst), worst_name)


def _percentiles(xs: List[float], ps: List[float]) -> Dict[str, float]:
    if not xs:
        return {f"p{int(p)}": float("nan") for p in ps}
    xs_sorted = sorted(xs)
    out: Dict[str, float] = {}
    for p in ps:
        if p <= 0:
            out[f"p{int(p)}"] = float(xs_sorted[0])
            continue
        if p >= 100:
            out[f"p{int(p)}"] = float(xs_sorted[-1])
            continue
        k = int(math.floor((p / 100.0) * (len(xs_sorted) - 1)))
        out[f"p{int(p)}"] = float(xs_sorted[k])
    return out


@dataclasses.dataclass
class FitRun:
    ok: bool
    fit_wall_s: float
    fit_cpu_s: float
    nll: float
    params: List[float]
    converged: Optional[bool] = None
    n_iter: Optional[int] = None
    n_fev: Optional[int] = None
    n_gev: Optional[int] = None
    n_eval: Optional[int] = None
    error: Optional[str] = None


@dataclasses.dataclass
class PreparedPyhf:
    model: Any
    data: Any
    names: List[str]


@dataclasses.dataclass
class PreparedNextStat:
    model: Any
    names: List[str]


def _prepare_pyhf(
    *,
    pyhf,
    ws_dict: dict[str, Any],
    measurement_name: str,
) -> PreparedPyhf:
    ws = pyhf.Workspace(ws_dict)
    model = ws.model(
        measurement_name=measurement_name,
        modifier_settings={
            "normsys": {"interpcode": "code4"},
            "histosys": {"interpcode": "code4p"},
        },
    )
    data = ws.data(model)
    names = list(model.config.par_names)
    return PreparedPyhf(model=model, data=data, names=names)


def _prepare_nextstat(*, nextstat, ws_dict: dict[str, Any]) -> PreparedNextStat:
    model = nextstat.HistFactoryModel.from_workspace(json.dumps(ws_dict))
    names = list(model.parameter_names())
    return PreparedNextStat(model=model, names=names)


def _fit_pyhf_prepared(
    *,
    pyhf,
    prepared: PreparedPyhf,
    fit_options: dict[str, Any],
    data_override: Any = None,
) -> FitRun:
    t0_wall = time.perf_counter()
    t0_cpu = time.process_time()
    try:
        data = prepared.data if data_override is None else data_override
        bestfit, twice_nll = pyhf.infer.mle.fit(data, prepared.model, return_fitted_val=True, **fit_options)
        nll = float(twice_nll.item()) / 2.0
        return FitRun(
            ok=True,
            fit_wall_s=float(time.perf_counter() - t0_wall),
            fit_cpu_s=float(time.process_time() - t0_cpu),
            nll=float(nll),
            params=[float(x) for x in list(bestfit)],
        )
    except Exception:
        return FitRun(
            ok=False,
            fit_wall_s=float(time.perf_counter() - t0_wall),
            fit_cpu_s=float(time.process_time() - t0_cpu),
            nll=float("nan"),
            params=[],
            error=traceback.format_exc(),
        )


def _fit_nextstat_prepared(
    *,
    prepared: PreparedNextStat,
    mle,
    main_data_override: Optional[List[float]] = None,
) -> FitRun:
    t0_wall = time.perf_counter()
    t0_cpu = time.process_time()
    try:
        res = mle.fit(prepared.model, data=main_data_override) if main_data_override is not None else mle.fit(prepared.model)
        return FitRun(
            ok=True,
            fit_wall_s=float(time.perf_counter() - t0_wall),
            fit_cpu_s=float(time.process_time() - t0_cpu),
            nll=float(res.nll),
            params=[float(x) for x in list(res.parameters)],
            converged=bool(getattr(res, "converged", True)),
            n_iter=int(getattr(res, "n_iter", 0)) if hasattr(res, "n_iter") else None,
            n_fev=int(getattr(res, "n_fev", 0)) if hasattr(res, "n_fev") else None,
            n_gev=int(getattr(res, "n_gev", 0)) if hasattr(res, "n_gev") else None,
            n_eval=int(getattr(res, "n_evaluations", 0)) if hasattr(res, "n_evaluations") else None,
        )
    except Exception:
        return FitRun(
            ok=False,
            fit_wall_s=float(time.perf_counter() - t0_wall),
            fit_cpu_s=float(time.process_time() - t0_cpu),
            nll=float("nan"),
            params=[],
            error=traceback.format_exc(),
        )


def _write_jsonl_line(fp, obj: Any) -> None:
    fp.write(json.dumps(obj, sort_keys=True) + "\n")
    fp.flush()


def _workspace_id(path: Path, ws: dict[str, Any]) -> str:
    # Lightweight stable id for publishable logs: file name + version + measurement.
    # (Avoid hashing huge JSON by default.)
    version = ws.get("version")
    meas = ws.get("measurements", [{}])[0].get("name")
    return f"{path.name}::v={version}::m={meas}"


def _summarize_runs(
    rows: List[dict[str, Any]],
    *,
    pyhf_names: List[str],
    nextstat_names: List[str],
    pyhf_build_s: float,
    nextstat_build_s: float,
    assessment_gates: Optional[dict[str, float]] = None,
) -> dict[str, Any]:
    # rows: only non-warmup, single workspace.
    #
    # Important: we treat pyhf as the reference objective, but we validate NextStat as a tool.
    # For publishable stability statistics, we compute parameter/NLL aggregates only on runs
    # where NextStat converged, and we pair pyhf rows by run index to avoid selection bias.
    pyhf_ok = [r for r in rows if r.get("pyhf", {}).get("ok")]
    ns_ok = [r for r in rows if r.get("nextstat", {}).get("ok")]

    paired_ok = [r for r in rows if r.get("pyhf", {}).get("ok") and r.get("nextstat", {}).get("ok")]
    paired_converged = [r for r in paired_ok if r.get("nextstat", {}).get("converged") is True]

    gates = assessment_gates or {}
    gate_obj_parity_p99 = float(gates.get("obj_parity_abs_p99", 1e-6))
    gate_nextstat_conv_rate_min = float(gates.get("nextstat_conv_rate_min", 0.95))
    gate_min_selected_runs = int(float(gates.get("min_selected_runs", 50)))

    def tool_stats(tool_rows: List[dict[str, Any]], tool_key: str) -> dict[str, Any]:
        wall = [float(r[tool_key]["fit_wall_s"]) for r in tool_rows]
        cpu = [float(r[tool_key]["fit_cpu_s"]) for r in tool_rows]
        nll = [float(r[tool_key]["nll"]) for r in tool_rows]
        out = {
            "n_ok": len(tool_rows),
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
        if tool_key == "nextstat" and tool_rows:
            conv = [bool(r[tool_key].get("converged")) for r in tool_rows if r[tool_key].get("converged") is not None]
            n_conv = sum(1 for c in conv if c)
            out["n_converged"] = int(n_conv)
            out["converged_rate"] = float(n_conv / len(conv)) if conv else float("nan")
        return out

    def param_stats(tool_rows: List[dict[str, Any]], *, names: List[str], tool_key: str) -> dict[str, Any]:
        if not tool_rows or not names:
            return {"names": names, "mean": [], "stdev": []}
        n = len(names)
        sums = [0.0] * n
        sumsq = [0.0] * n
        used = 0
        for r in tool_rows:
            ps = r[tool_key].get("params")
            if not isinstance(ps, list) or len(ps) != n:
                continue
            used += 1
            for i, v in enumerate(ps):
                fv = float(v)
                sums[i] += fv
                sumsq[i] += fv * fv
        if used == 0:
            return {"names": names, "mean": [float("nan")] * n, "stdev": [float("nan")] * n, "n_used": 0}
        means = [s / used for s in sums]
        stdevs: List[float] = []
        for i in range(n):
            m = means[i]
            var = max((sumsq[i] / used) - (m * m), 0.0)
            stdevs.append(float(math.sqrt(var)))
        return {"names": names, "mean": means, "stdev": stdevs, "n_used": used}

    def cross_eval_stats(rows_for_eval: List[dict[str, Any]]) -> dict[str, Any]:
        # Cross-eval validates "same objective" even if optimizers land elsewhere.
        cross_pyhf_at_ns = [
            float(r.get("cross_eval", {}).get("pyhf_nll_at_nextstat_hat"))
            for r in rows_for_eval
            if r.get("cross_eval", {}).get("pyhf_nll_at_nextstat_hat") is not None
        ]
        cross_ns_at_pyhf = [
            float(r.get("cross_eval", {}).get("nextstat_nll_at_pyhf_hat"))
            for r in rows_for_eval
            if r.get("cross_eval", {}).get("nextstat_nll_at_pyhf_hat") is not None
        ]
        return {
            "pyhf_nll_at_nextstat_hat": {
                "mean": float(statistics.mean(cross_pyhf_at_ns)) if cross_pyhf_at_ns else float("nan"),
                "stdev": float(statistics.pstdev(cross_pyhf_at_ns)) if len(cross_pyhf_at_ns) >= 2 else 0.0,
                "n": int(len(cross_pyhf_at_ns)),
            },
            "nextstat_nll_at_pyhf_hat": {
                "mean": float(statistics.mean(cross_ns_at_pyhf)) if cross_ns_at_pyhf else float("nan"),
                "stdev": float(statistics.pstdev(cross_ns_at_pyhf)) if len(cross_ns_at_pyhf) >= 2 else 0.0,
                "n": int(len(cross_ns_at_pyhf)),
            },
        }

    cross_eval_all_ok = cross_eval_stats(paired_ok)
    cross_eval_converged = cross_eval_stats(paired_converged)

    def dist(vals: List[float]) -> dict[str, Any]:
        out: dict[str, Any] = {
            "n": int(len(vals)),
            "mean": float(statistics.mean(vals)) if vals else float("nan"),
            "stdev": float(statistics.pstdev(vals)) if len(vals) >= 2 else 0.0,
            "min": float(min(vals)) if vals else float("nan"),
            "max": float(max(vals)) if vals else float("nan"),
        }
        out.update(_percentiles(vals, [50, 90, 99]))
        return out

    # Assessment:
    # - "Objective parity" checks whether both toolchains compute (nearly) the same NLL on the
    #   same parameters and data (independent of optimizer performance).
    # - "Optimizer agreement" compares the fitted minima NLLs and indicates which optimizer
    #   tends to find a lower minimum under the shared objective.
    obj_parity_pyhf_at_ns_abs: List[float] = []
    obj_parity_ns_at_pyhf_abs: List[float] = []
    abs_delta_hat_nll: List[float] = []
    pyhf_improvement_from_ns: List[float] = []
    ns_improvement_from_pyhf: List[float] = []
    for r in paired_ok:
        try:
            py_nll = float(r["pyhf"]["nll"])
            ns_nll = float(r["nextstat"]["nll"])
            cross_pyhf_at_ns = float(r.get("cross_eval", {}).get("pyhf_nll_at_nextstat_hat"))
            cross_ns_at_pyhf = float(r.get("cross_eval", {}).get("nextstat_nll_at_pyhf_hat"))
        except Exception:
            continue

        if all(map(math.isfinite, [py_nll, ns_nll, cross_pyhf_at_ns, cross_ns_at_pyhf])):
            obj_parity_pyhf_at_ns_abs.append(abs(cross_pyhf_at_ns - ns_nll))
            obj_parity_ns_at_pyhf_abs.append(abs(cross_ns_at_pyhf - py_nll))
            abs_delta_hat_nll.append(abs(ns_nll - py_nll))

            # Positive => the other tool's optimum is better under this tool's objective.
            pyhf_improvement_from_ns.append(py_nll - cross_pyhf_at_ns)
            ns_improvement_from_pyhf.append(ns_nll - cross_ns_at_pyhf)

    # NextStat convergence rate among paired-ok runs (not only selected-converged).
    n_paired_ok = len(paired_ok)
    conv_rate = (len(paired_converged) / n_paired_ok) if n_paired_ok > 0 else 0.0

    # Determine PASS/FAIL/INCONCLUSIVE (objective parity is the hard correctness gate).
    status = "INCONCLUSIVE"
    reasons: List[str] = []
    if n_paired_ok == 0:
        reasons.append("no paired-ok runs")
    else:
        if conv_rate < gate_nextstat_conv_rate_min:
            reasons.append(f"nextstat conv_rate={conv_rate:.3g} < {gate_nextstat_conv_rate_min:.3g}")
        if len(paired_converged) < gate_min_selected_runs:
            reasons.append(f"selected_n={len(paired_converged)} < {gate_min_selected_runs}")

        p99_pyhf_at_ns = float(dist(obj_parity_pyhf_at_ns_abs).get("p99", float("nan")))
        p99_ns_at_pyhf = float(dist(obj_parity_ns_at_pyhf_abs).get("p99", float("nan")))
        if math.isfinite(p99_pyhf_at_ns) and p99_pyhf_at_ns > gate_obj_parity_p99:
            reasons.append(f"obj_parity(pyhf@ns_hat) p99={p99_pyhf_at_ns:.3g} > {gate_obj_parity_p99:.3g}")
        if math.isfinite(p99_ns_at_pyhf) and p99_ns_at_pyhf > gate_obj_parity_p99:
            reasons.append(f"obj_parity(ns@pyhf_hat) p99={p99_ns_at_pyhf:.3g} > {gate_obj_parity_p99:.3g}")

        if any("obj_parity" in r for r in reasons):
            status = "FAIL"
        elif not reasons:
            status = "PASS"
        else:
            status = "INCONCLUSIVE"

    assessment = {
        "status": status,
        "reasons": reasons,
        "gates": {
            "obj_parity_abs_p99": gate_obj_parity_p99,
            "nextstat_conv_rate_min": gate_nextstat_conv_rate_min,
            "min_selected_runs": gate_min_selected_runs,
        },
        "counts": {
            "n_total_runs": int(len(rows)),
            "n_paired_ok": int(n_paired_ok),
            "n_selected_converged": int(len(paired_converged)),
        },
        "rates": {"nextstat_conv_rate": float(conv_rate)},
        "objective_parity": {
            "abs(pyhf_nll_at_nextstat_hat - nextstat_nll)": dist(obj_parity_pyhf_at_ns_abs),
            "abs(nextstat_nll_at_pyhf_hat - pyhf_nll)": dist(obj_parity_ns_at_pyhf_abs),
        },
        "optimizer": {
            "abs(nextstat_hat_nll - pyhf_hat_nll)": dist(abs_delta_hat_nll),
            "pyhf_improvement_from_nextstat_hat": dist(pyhf_improvement_from_ns),
            "nextstat_improvement_from_pyhf_hat": dist(ns_improvement_from_pyhf),
        },
    }

    # Name-aligned mean diffs (in pyhf order), using converged-only paired runs.
    py_params = param_stats(paired_converged, names=pyhf_names, tool_key="pyhf")
    ns_params = param_stats(paired_converged, names=nextstat_names, tool_key="nextstat")

    # Also keep all-ok stats for transparency (timing and convergence failures matter for tooling).
    py_all = tool_stats(pyhf_ok, "pyhf")
    ns_all = tool_stats(ns_ok, "nextstat")

    # Selected (publishable) aggregates: only NextStat-converged paired runs.
    py_sel = tool_stats(paired_converged, "pyhf")
    ns_sel = tool_stats(paired_converged, "nextstat")

    selection = {
        "n_total_runs": int(len(rows)),
        "n_pyhf_ok": int(len(pyhf_ok)),
        "n_nextstat_ok": int(len(ns_ok)),
        "n_paired_ok": int(len(paired_ok)),
        "n_nextstat_converged": int(len(paired_converged)),
    }

    ns_index = {n: i for i, n in enumerate(nextstat_names)}
    ns_mean_in_pyhf: List[float] = []
    ns_stdev_in_pyhf: List[float] = []
    abs_mean_diff: List[float] = []
    for i, name in enumerate(pyhf_names):
        j = ns_index.get(name)
        if j is None or j >= len(ns_params["mean"]):
            ns_mean_in_pyhf.append(float("nan"))
            ns_stdev_in_pyhf.append(float("nan"))
            abs_mean_diff.append(float("nan"))
            continue
        nm = float(ns_params["mean"][j])
        nsd = float(ns_params["stdev"][j])
        pm = float(py_params["mean"][i]) if i < len(py_params["mean"]) else float("nan")
        ns_mean_in_pyhf.append(nm)
        ns_stdev_in_pyhf.append(nsd)
        abs_mean_diff.append(abs(pm - nm) if (math.isfinite(pm) and math.isfinite(nm)) else float("nan"))

    rows_mean = []
    for name, d in zip(pyhf_names, abs_mean_diff):
        if math.isfinite(float(d)):
            rows_mean.append((float(d), name))
    rows_mean.sort(reverse=True)

    return {
        "selection": selection,
        "build_s": {"pyhf": float(pyhf_build_s), "nextstat": float(nextstat_build_s)},
        "pyhf": py_sel,
        "nextstat": ns_sel,
        "all_ok": {"pyhf": py_all, "nextstat": ns_all},
        "cross_eval": {"all_ok": cross_eval_all_ok, "converged": cross_eval_converged},
        "assessment": assessment,
        "params": {
            "pyhf": py_params,
            "nextstat": ns_params,
            "nextstat_in_pyhf_order": {
                "names": pyhf_names,
                "mean": ns_mean_in_pyhf,
                "stdev": ns_stdev_in_pyhf,
            },
            "abs_mean_diff_pyhf_order": {
                "names": pyhf_names,
                "abs_diff": abs_mean_diff,
                "top": [{"name": n, "abs_diff": float(d)} for d, n in rows_mean[:20]],
            },
        },
    }


def _write_summary(
    *,
    out_json: Path,
    out_md: Path,
    header: dict[str, Any],
    records: List[dict[str, Any]],
    exclude_workspace_ids: List[str],
    assessment_gates: dict[str, float],
) -> None:
    # Group by workspace id, then compute summary stats.
    by_ws: Dict[str, List[dict[str, Any]]] = {}
    metas: Dict[str, dict[str, Any]] = {}
    for r in records:
        if r.get("type") == "workspace_meta":
            metas[str(r["workspace_id"])] = r
            continue
        if r.get("type") == "run":
            by_ws.setdefault(str(r["workspace_id"]), []).append(r)

    ws_summaries: List[dict[str, Any]] = []
    for ws_id, rows in sorted(by_ws.items(), key=lambda kv: kv[0]):
        if ws_id in exclude_workspace_ids:
            continue
        rows = [r for r in rows if not r.get("warmup")]
        meta = metas.get(ws_id, {})
        pyhf_names = list(meta.get("pyhf", {}).get("names") or [])
        ns_names = list(meta.get("nextstat", {}).get("names") or [])
        pyhf_build_s = float(meta.get("pyhf", {}).get("build_s", float("nan")))
        ns_build_s = float(meta.get("nextstat", {}).get("build_s", float("nan")))
        ws_summaries.append(
            {
                "workspace_id": ws_id,
                "path": rows[0].get("path") if rows else None,
                "n_runs": len(rows),
                "summary": _summarize_runs(
                    rows,
                    pyhf_names=pyhf_names,
                    nextstat_names=ns_names,
                    pyhf_build_s=pyhf_build_s,
                    nextstat_build_s=ns_build_s,
                    assessment_gates=assessment_gates,
                ),
            }
        )

    report = {"header": header, "workspaces": ws_summaries}
    out_json.parent.mkdir(parents=True, exist_ok=True)
    out_json.write_text(json.dumps(report, indent=2, sort_keys=True) + "\n")

    # Markdown: keep short; raw data lives in JSONL.
    lines: List[str] = []
    lines.append("# Repeated MLE fits (pyhf vs NextStat)")
    lines.append("")
    lines.append(f"Generated: `{time.strftime('%Y-%m-%d %H:%M:%S')}`")
    lines.append("")
    lines.append("## Config")
    lines.append("")
    lines.append(f"- n_runs: {header.get('n_runs')}, n_warmup: {header.get('n_warmup')}")
    lines.append(f"- excluded: {header.get('excluded')}")
    lines.append(f"- pyhf_fit_options: `{header.get('pyhf_fit_options')}`")
    lines.append(f"- nextstat_mle_config: `{header.get('nextstat_mle_config')}`")
    lines.append("")
    lines.append("## Results (timing + NLL)")
    lines.append("")
    lines.append("| Workspace | pyhf fit mean/p50/p90 (s) | NextStat fit mean/p50/p90 (s) | Speedup (mean) | pyhf nll mean | NextStat nll mean | Notes |")
    lines.append("|---|---:|---:|---:|---:|---:|---|")
    for ws in ws_summaries:
        s = ws["summary"]
        py = s["pyhf"]  # selected: NextStat-converged paired runs
        ns = s["nextstat"]  # selected: NextStat-converged paired runs
        sel = s.get("selection", {})
        all_ok = s.get("all_ok", {})
        py_all = all_ok.get("pyhf", {})
        ns_all = all_ok.get("nextstat", {})
        py_t = float(py["fit_wall_s"]["mean"])
        ns_t = float(ns["fit_wall_s"]["mean"])
        speed = (py_t / ns_t) if (math.isfinite(py_t) and math.isfinite(ns_t) and ns_t > 0) else float("nan")
        py_nll = float(py["nll"]["mean"])
        ns_nll = float(ns["nll"]["mean"])
        py_p50 = float(py["fit_wall_s"]["p50"])
        py_p90 = float(py["fit_wall_s"]["p90"])
        ns_p50 = float(ns["fit_wall_s"]["p50"])
        ns_p90 = float(ns["fit_wall_s"]["p90"])
        notes = ""
        try:
            n_total = int(sel.get("n_total_runs", 0))
            n_ok_ns = int(sel.get("n_nextstat_ok", 0))
            n_sel = int(sel.get("n_nextstat_converged", 0))
            conv_rate = float(ns_all.get("converged_rate", float("nan")))
            notes = f"selected={n_sel}/{n_total}; conv={n_sel}/{n_ok_ns} (rate={conv_rate:.3g})"
        except Exception:
            pass
        if math.isfinite(py_nll) and math.isfinite(ns_nll):
            if ns_nll < py_nll:
                notes = (notes + "; " if notes else "") + "nextstat nll < pyhf"
            elif py_nll < ns_nll:
                notes = (notes + "; " if notes else "") + "pyhf nll < nextstat"

        # Show all-ok speedup too (includes non-converged cost, if any).
        try:
            py_all_t = float(((py_all.get("fit_wall_s") or {}).get("mean")))
            ns_all_t = float(((ns_all.get("fit_wall_s") or {}).get("mean")))
            speed_all = (py_all_t / ns_all_t) if (math.isfinite(py_all_t) and math.isfinite(ns_all_t) and ns_all_t > 0) else float("nan")
            if math.isfinite(speed_all):
                notes = (notes + "; " if notes else "") + f"speedup_all_ok={speed_all:.2f}x"
        except Exception:
            pass
        lines.append(
            f"| `{ws['path']}` | {py_t:.6f}/{py_p50:.6f}/{py_p90:.6f} | {ns_t:.6f}/{ns_p50:.6f}/{ns_p90:.6f} | {speed:.2f}x | {py_nll:.6g} | {ns_nll:.6g} | {notes} |"
        )

    # Include top mean-diff params for the most divergent workspaces.
    worst = []
    for ws in ws_summaries:
        top = ws["summary"].get("params", {}).get("abs_mean_diff_pyhf_order", {}).get("top") or []
        if top:
            worst.append((float(top[0]["abs_diff"]), ws, top))
    worst.sort(reverse=True, key=lambda t: t[0])
    for _, ws, top in worst[:5]:
        lines.append("")
        lines.append(f"### Mean best-fit diffs (top 10): `{ws['path']}`")
        lines.append("")
        for row in top[:10]:
            lines.append(f"- `{row['name']}`: |Δmean|={float(row['abs_diff']):.6g}")

    # Add a short correctness assessment section (objective parity gates).
    lines.append("")
    lines.append("## Assessment")
    lines.append("")
    for ws in ws_summaries:
        a = (ws.get("summary") or {}).get("assessment") or {}
        status = str(a.get("status") or "INCONCLUSIVE")
        conv_rate = (a.get("rates") or {}).get("nextstat_conv_rate")
        p1 = (a.get("objective_parity") or {}).get("abs(pyhf_nll_at_nextstat_hat - nextstat_nll)") or {}
        p2 = (a.get("objective_parity") or {}).get("abs(nextstat_nll_at_pyhf_hat - pyhf_nll)") or {}
        reasons = a.get("reasons") or []
        lines.append(f"### `{ws['path']}`")
        lines.append("")
        lines.append(f"- status: `{status}`")
        if conv_rate is not None:
            lines.append(f"- nextstat conv_rate: {float(conv_rate):.6g}")
        if p1.get("n", 0) or p2.get("n", 0):
            lines.append(
                f"- obj_parity abs p99: pyhf@ns_hat={float(p1.get('p99', float('nan'))):.6g}, ns@pyhf_hat={float(p2.get('p99', float('nan'))):.6g}"
            )
            lines.append(
                f"- obj_parity abs max: pyhf@ns_hat={float(p1.get('max', float('nan'))):.6g}, ns@pyhf_hat={float(p2.get('max', float('nan'))):.6g}"
            )
        if reasons:
            lines.append(f"- reasons: `{'; '.join(str(x) for x in reasons)}`")

    out_md.parent.mkdir(parents=True, exist_ok=True)
    out_md.write_text("\n".join(lines) + "\n")


def main(argv: Optional[List[str]] = None) -> int:
    ap = argparse.ArgumentParser()
    ap.add_argument("--workspace", action="append", default=[], help="Workspace JSON path (repeatable). Default: tests/fixtures/*.json")
    ap.add_argument("--exclude", action="append", default=[], help="Exclude if substring matches path (repeatable).")
    ap.add_argument("--n-runs", type=int, default=3)
    ap.add_argument("--n-warmup", type=int, default=1, help="Warmup runs per workspace (not counted in summary).")
    ap.add_argument("--out-jsonl", type=Path, default=Path("tmp/repeat_mle_fits.jsonl"))
    ap.add_argument("--out-summary-json", type=Path, default=Path("tmp/repeat_mle_fits_summary.json"))
    ap.add_argument("--out-summary-md", type=Path, default=Path("tmp/repeat_mle_fits_summary.md"))
    ap.add_argument("--resume", action="store_true", help="Append to existing JSONL and continue missing runs (requires same config).")
    ap.add_argument("--threads", type=int, default=1, help="Set thread env vars (RAYON/OMP/MKL/OPENBLAS/VECLIB).")
    ap.add_argument("--toy-main", action="store_true", help="Poisson-fluctuate main observations per run (auxdata fixed).")
    ap.add_argument("--toy-seed", type=int, default=123, help="Base RNG seed for toy-main generation.")
    ap.add_argument("--gate-obj-parity-abs-p99", type=float, default=1e-6, help="Gate: p99(|pyhf_nll_at_ns_hat - ns_nll|, |ns_nll_at_pyhf_hat - pyhf_nll|) <= threshold.")
    ap.add_argument("--gate-nextstat-conv-rate-min", type=float, default=0.95, help="Gate: min NextStat convergence rate among paired-ok runs (else INCONCLUSIVE).")
    ap.add_argument("--gate-min-selected-runs", type=int, default=50, help="Gate: require at least this many NextStat-converged paired runs (else INCONCLUSIVE).")
    ap.add_argument(
        "--fit-order",
        choices=["pyhf-first", "nextstat-first", "alternate"],
        default="pyhf-first",
        help="Order of tool execution per run. Use 'alternate' to reduce systematic cache/CPU-state bias in timing comparisons.",
    )
    ap.add_argument("--pyhf-method", default="SLSQP")
    ap.add_argument("--pyhf-maxiter", type=int, default=100000)
    ap.add_argument("--pyhf-tolerance", type=float, default=float("nan"))
    ap.add_argument("--pyhf-do-grad", type=int, default=-1, help="Set to 0/1 to override pyhf do_grad, or -1 to keep default.")
    ap.add_argument("--pyhf-do-stitch", type=int, default=0, help="Set to 1 to enable pyhf do_stitch (helps some SciPy methods like L-BFGS-B).")
    ap.add_argument("--nextstat-max-iter", type=int, default=1000)
    ap.add_argument("--nextstat-tol", type=float, default=1e-6)
    ap.add_argument("--nextstat-m", type=int, default=10)
    args = ap.parse_args(argv)

    if int(args.n_runs) <= 0:
        raise SystemExit("--n-runs must be >= 1")
    if int(args.n_warmup) < 0:
        raise SystemExit("--n-warmup must be >= 0")

    # Determinism / fairness knobs.
    threads = int(args.threads)
    if threads >= 1:
        for k in [
            "RAYON_NUM_THREADS",
            "OMP_NUM_THREADS",
            "MKL_NUM_THREADS",
            "OPENBLAS_NUM_THREADS",
            "VECLIB_MAXIMUM_THREADS",
            "NUMEXPR_NUM_THREADS",
        ]:
            os.environ[k] = str(threads)

    import nextstat  # noqa: WPS433

    pyhf = None
    try:
        import pyhf as _pyhf  # noqa: WPS433

        pyhf = _pyhf
    except Exception:
        raise SystemExit("pyhf is required. Install: pip install -e 'bindings/ns-py[validation]'")

    # Guard against a common misconfiguration:
    # - With the default numpy backend, pyhf cannot compute gradients (no autodiff),
    #   so do_grad=True will hard-fail.
    if args.pyhf_do_grad == 1:
        tensorlib, _ = pyhf.get_backend()
        if getattr(tensorlib, "name", None) == "numpy":
            raise SystemExit(
                "pyhf backend is numpy, which does not support autodifferentiation. "
                "Use --pyhf-do-grad 0 (recommended for numpy), or install an autodiff backend "
                "(jax/tensorflow/torch) and set it via pyhf.set_backend(...)."
            )

    pyhf_fit_options: dict[str, Any] = {"method": str(args.pyhf_method), "maxiter": int(args.pyhf_maxiter)}
    if math.isfinite(float(args.pyhf_tolerance)):
        pyhf_fit_options["tolerance"] = float(args.pyhf_tolerance)
    if args.pyhf_do_grad in (0, 1):
        pyhf_fit_options["do_grad"] = bool(args.pyhf_do_grad)
    if args.pyhf_do_stitch in (0, 1):
        pyhf_fit_options["do_stitch"] = bool(args.pyhf_do_stitch)

    nextstat_mle_config = {"max_iter": int(args.nextstat_max_iter), "tol": float(args.nextstat_tol), "m": int(args.nextstat_m)}
    assessment_gates = {
        "obj_parity_abs_p99": float(args.gate_obj_parity_abs_p99),
        "nextstat_conv_rate_min": float(args.gate_nextstat_conv_rate_min),
        "min_selected_runs": float(int(args.gate_min_selected_runs)),
    }

    paths = [Path(p) for p in args.workspace] if args.workspace else list(_iter_default_workspace_paths())
    # Keep historical default exclusions only when running the whole fixture suite.
    excludes_raw = list(args.exclude)
    if not args.workspace:
        excludes_raw += ["tchannel_workspace.json"]
    excludes: List[str] = []
    for x in excludes_raw:
        if x not in excludes:
            excludes.append(x)

    header = {
        "generated_at": time.strftime("%Y-%m-%d %H:%M:%S"),
        "python": sys.version,
        "platform": platform.platform(),
        "machine": platform.machine(),
        "processor": platform.processor(),
        **_git_meta(),
        "pyhf_version": getattr(pyhf, "__version__", "unknown"),
        "pyhf_backend": getattr(pyhf.get_backend()[0], "name", "unknown"),
        "pyhf_optimizer": pyhf.get_backend()[1].__class__.__name__,
        "nextstat_version": getattr(nextstat, "__version__", "unknown"),
        "n_runs": int(args.n_runs),
        "n_warmup": int(args.n_warmup),
        "threads": threads,
        "toy_main": bool(args.toy_main),
        "toy_seed": int(args.toy_seed),
        "fit_order": str(args.fit_order),
        "excluded": excludes,
        "pyhf_fit_options": pyhf_fit_options,
        "nextstat_mle_config": nextstat_mle_config,
        "assessment_gates": assessment_gates,
    }

    args.out_jsonl.parent.mkdir(parents=True, exist_ok=True)
    records: List[dict[str, Any]] = []
    excluded_ws_ids: List[str] = []

    # Resume support: load existing records and keep header immutable.
    existing: List[dict[str, Any]] = []
    if bool(args.resume) and args.out_jsonl.exists():
        existing = _load_jsonl_records(args.out_jsonl)
        if existing and existing[0].get("type") == "header":
            prev = existing[0]
            # Hard guard: do not mix configurations in one log.
            must_match = [
                "n_runs",
                "n_warmup",
                "threads",
                "toy_main",
                "toy_seed",
                "fit_order",
                "pyhf_fit_options",
                "nextstat_mle_config",
                "assessment_gates",
            ]
            mism = []
            for k in must_match:
                if prev.get(k) != header.get(k):
                    mism.append(k)
            if mism:
                raise SystemExit(f"--resume requested but header config differs for keys: {mism}")
            header = prev
        records.extend([r for r in existing if isinstance(r, dict)])

    mode = "a" if (bool(args.resume) and args.out_jsonl.exists()) else "w"
    with args.out_jsonl.open(mode, encoding="utf-8") as fp:
        if mode == "w":
            _write_jsonl_line(fp, {"type": "header", **header})

        for p in paths:
            if any(ex in str(p) for ex in excludes):
                continue

            try:
                t_load0 = time.perf_counter()
                ws = _load_json(p)
                load_s = float(time.perf_counter() - t_load0)
            except Exception:
                err = traceback.format_exc()
                rec = {"type": "workspace_error", "path": str(p), "where": "load_json", "error": err}
                _write_jsonl_line(fp, rec)
                continue

            if not _is_pyhf_workspace(ws):
                excluded_ws_ids.append(_workspace_id(p, ws) if isinstance(ws, dict) else p.name)
                rec = {"type": "skipped", "path": str(p), "reason": "not_pyhf_workspace"}
                _write_jsonl_line(fp, rec)
                continue

            try:
                meas = _measurement_name(ws)
            except Exception:
                excluded_ws_ids.append(_workspace_id(p, ws))
                rec = {"type": "skipped", "path": str(p), "reason": "no_measurement_name"}
                _write_jsonl_line(fp, rec)
                continue

            ws_id = _workspace_id(p, ws)
            print(f"[repeat] {p} ({ws_id})", flush=True)

            # Build models once per workspace (exclude build from per-fit timing).
            try:
                t_py0 = time.perf_counter()
                prep_py = _prepare_pyhf(pyhf=pyhf, ws_dict=ws, measurement_name=meas)
                pyhf_build_s = float(time.perf_counter() - t_py0)
            except Exception:
                err = traceback.format_exc()
                rec = {"type": "workspace_error", "path": str(p), "where": "pyhf_prepare", "error": err}
                _write_jsonl_line(fp, rec)
                continue

            try:
                t_ns0 = time.perf_counter()
                prep_ns = _prepare_nextstat(nextstat=nextstat, ws_dict=ws)
                nextstat_build_s = float(time.perf_counter() - t_ns0)
            except Exception:
                err = traceback.format_exc()
                rec = {"type": "workspace_error", "path": str(p), "where": "nextstat_prepare", "error": err}
                _write_jsonl_line(fp, rec)
                continue

            meta_rec = {
                "type": "workspace_meta",
                "workspace_id": ws_id,
                "path": str(p),
                "measurement": meas,
                "load_json_s": load_s,
                "pyhf": {"build_s": pyhf_build_s, "n_params": len(prep_py.names), "names": prep_py.names},
                "nextstat": {
                    "build_s": nextstat_build_s,
                    "n_params": len(prep_ns.names),
                    "names": prep_ns.names,
                },
            }
            already_has_meta = any(
                (r.get("type") == "workspace_meta") and (str(r.get("workspace_id")) == ws_id) for r in records
            )
            if not (bool(args.resume) and already_has_meta):
                _write_jsonl_line(fp, meta_rec)
                records.append(meta_rec)

            # Warmup + measured runs.
            mle = nextstat.MaximumLikelihoodEstimator(**nextstat_mle_config)

            base_data = None
            n_main = None
            exp_main = None
            seed_ws = None
            if bool(args.toy_main):
                try:
                    # Determine main/aux split from the actual built model (pyhf order).
                    n_data = len(prep_py.data)
                    n_aux = len(prep_py.model.config.auxdata)
                    n_main = int(n_data - n_aux)
                    if n_main <= 0:
                        raise ValueError(f"invalid n_main={n_main} (n_data={n_data}, n_aux={n_aux})")
                    base_data = pyhf.tensorlib.tolist(prep_py.data)

                    # Truth point: observed-data MLE in pyhf.
                    truth_fit = _fit_pyhf_prepared(pyhf=pyhf, prepared=prep_py, fit_options=pyhf_fit_options)
                    if not truth_fit.ok:
                        raise RuntimeError(f"pyhf truth fit failed: {truth_fit.error}")

                    truth_pars = truth_fit.params
                    exp_all = prep_py.model.expected_data(truth_pars, include_auxdata=True)
                    exp_all = [float(x) for x in list(pyhf.tensorlib.tolist(exp_all))]
                    exp_main = exp_all[:n_main]
                    if len(exp_main) != n_main:
                        raise ValueError("expected_main length mismatch")
                    if any((not math.isfinite(x)) or x < 0.0 for x in exp_main):
                        raise ValueError("expected_main contains non-finite/negative values")

                    # Stable seed per workspace (metadata only). Each run uses a per-run seed
                    # derived from (ws_id, toy_seed, run_idx) to make resume safe.
                    h = hashlib.sha256(ws_id.encode("utf-8")).digest()
                    seed_ws = (int.from_bytes(h[:8], "little") ^ int(args.toy_seed)) & 0x7FFF_FFFF_FFFF_FFFF

                    # Attach toy metadata to the workspace meta record.
                    meta_rec["toy"] = {
                        "n_main": int(n_main),
                        "n_aux": int(n_aux),
                        "seed_workspace": int(seed_ws),
                        "truth_pars_source": "pyhf_observed_mle",
                    }
                    meta_rec["toy"]["truth_pars"] = truth_pars
                except Exception:
                    err = traceback.format_exc()
                    rec = {"type": "workspace_error", "path": str(p), "where": "toy_prepare", "error": err}
                    _write_jsonl_line(fp, rec)
                    continue

            start_idx = _next_run_idx(records, ws_id) if bool(args.resume) else 0
            for run_idx in range(int(start_idx), int(args.n_warmup) + int(args.n_runs)):
                is_warmup = run_idx < int(args.n_warmup)

                toy = None
                data_override = None
                main_override = None
                ns_model_for_eval = prep_ns.model

                if bool(args.toy_main):
                    assert base_data is not None and n_main is not None and exp_main is not None and seed_ws is not None
                    seed_run = _toy_seed_for_run(ws_id, int(args.toy_seed), int(run_idx))
                    rng = np.random.default_rng(seed_run)
                    # Poisson toys for main observations (auxdata stays fixed).
                    y_main = rng.poisson(lam=np.asarray(exp_main, dtype=float)).astype(float).tolist()
                    main_override = [float(x) for x in y_main]
                    data_override = list(base_data)
                    data_override[:n_main] = main_override
                    toy = {
                        "seed_workspace": int(seed_ws),
                        "seed_run": int(seed_run),
                        "main_hash_sha256": hashlib.sha256(np.asarray(main_override, dtype=np.float64).tobytes()).hexdigest(),
                    }
                    # For cross-eval on this toy dataset, we need a NextStat model with overridden main obs.
                    try:
                        ns_model_for_eval = prep_ns.model.with_observed_main(main_override)
                    except Exception:
                        ns_model_for_eval = prep_ns.model

                if str(args.fit_order) == "alternate":
                    pyhf_first = (int(run_idx) % 2) == 0
                else:
                    pyhf_first = str(args.fit_order) == "pyhf-first"

                if pyhf_first:
                    py_run = _fit_pyhf_prepared(
                        pyhf=pyhf,
                        prepared=prep_py,
                        fit_options=pyhf_fit_options,
                        data_override=data_override,
                    )
                    ns_run = _fit_nextstat_prepared(prepared=prep_ns, mle=mle, main_data_override=main_override)
                    fit_order_run = "pyhf-first"
                else:
                    ns_run = _fit_nextstat_prepared(prepared=prep_ns, mle=mle, main_data_override=main_override)
                    py_run = _fit_pyhf_prepared(
                        pyhf=pyhf,
                        prepared=prep_py,
                        fit_options=pyhf_fit_options,
                        data_override=data_override,
                    )
                    fit_order_run = "nextstat-first"

                diffs: Dict[str, Any] = {}
                py_names = prep_py.names
                ns_names = prep_ns.names
                if py_run.ok and ns_run.ok and py_names and ns_names and len(py_run.params) == len(py_names) and len(ns_run.params) == len(ns_names):
                    diffs["param_set_equal"] = bool(set(py_names) == set(ns_names))
                    max_d, worst_name = _max_abs_diff_by_name(py_names, py_run.params, ns_names, ns_run.params)
                    diffs["bestfit_max_abs_diff"] = float(max_d)
                    diffs["worst_param"] = worst_name
                    diffs["nll_delta"] = float(ns_run.nll - py_run.nll) if (math.isfinite(py_run.nll) and math.isfinite(ns_run.nll)) else float("nan")

                cross_eval: Dict[str, Any] = {}
                if py_run.ok and ns_run.ok and diffs.get("param_set_equal") is True:
                    try:
                        ns_in_pyhf = _remap_params(ns_names, ns_run.params, py_names)
                        data_for_eval = prep_py.data if data_override is None else data_override
                        pyhf_nll_at_ns = float(pyhf.infer.mle.twice_nll(ns_in_pyhf, data_for_eval, prep_py.model).item()) / 2.0
                        py_in_ns = _remap_params(py_names, py_run.params, ns_names)
                        ns_nll_at_py = float(ns_model_for_eval.nll(py_in_ns))
                        cross_eval = {
                            "pyhf_nll_at_nextstat_hat": float(pyhf_nll_at_ns),
                            "nextstat_nll_at_pyhf_hat": float(ns_nll_at_py),
                        }
                    except Exception:
                        cross_eval = {"error": traceback.format_exc()}

                rec = {
                    "type": "run",
                    "workspace_id": ws_id,
                    "path": str(p),
                    "measurement": meas,
                    "warmup": bool(is_warmup),
                    "run_idx": int(run_idx),
                    "fit_order_run": str(fit_order_run),
                    "toy": toy,
                    "pyhf": dataclasses.asdict(py_run),
                    "nextstat": dataclasses.asdict(ns_run),
                    "diffs": diffs,
                    "cross_eval": cross_eval,
                }
                _write_jsonl_line(fp, rec)
                records.append(rec)

    _write_summary(
        out_json=args.out_summary_json,
        out_md=args.out_summary_md,
        header=header,
        records=records,
        exclude_workspace_ids=excluded_ws_ids,
        assessment_gates=assessment_gates,
    )

    print(f"Wrote: {args.out_jsonl}")
    print(f"Wrote: {args.out_summary_json}")
    print(f"Wrote: {args.out_summary_md}")
    return 0


if __name__ == "__main__":
    raise SystemExit(main())
