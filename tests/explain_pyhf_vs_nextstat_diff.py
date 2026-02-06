#!/usr/bin/env python3
"""Explain pyhf vs NextStat differences at a parameter point.

This tool is for debugging mismatches. It computes:
  - NLL(pyhf) vs NLL(nextstat)
  - expected_data(pyhf) vs expected_data(nextstat) for main bins
  - top bins by abs(expected_data diff)
  - top parameters by sensitivity of (ns_nll - pyhf_nll) using finite differences

It also writes a JSON artifact with all details.

Run:
  PYTHONPATH=bindings/ns-py/python ./.venv/bin/python tests/explain_pyhf_vs_nextstat_diff.py \\
    --workspace tests/fixtures/complex_workspace.json --measurement measurement
"""

from __future__ import annotations

import argparse
import json
import math
import time
from pathlib import Path
from typing import Any, Dict, List, Tuple

import pyhf

import nextstat


def pyhf_model_and_data(workspace: dict[str, Any], measurement_name: str):
    ws = pyhf.Workspace(workspace)
    model = ws.model(
        measurement_name=measurement_name,
        modifier_settings={
            "normsys": {"interpcode": "code4"},
            "histosys": {"interpcode": "code4p"},
        },
    )
    data = ws.data(model)
    return model, data


def pyhf_nll(model, data, params) -> float:
    return float(pyhf.infer.mle.twice_nll(params, data, model).item()) / 2.0


def map_params_by_name(src_names, src_params, dst_names, dst_init):
    dst_index = {name: i for i, name in enumerate(dst_names)}
    out = list(dst_init)
    for name, value in zip(src_names, src_params):
        out[dst_index[name]] = float(value)
    return out


def finite_diff_sensitivity(
    f,
    x: List[float],
    bounds: List[Tuple[float, float]],
    *,
    rel_step: float = 1e-5,
    abs_step: float = 1e-6,
) -> List[float]:
    g: List[float] = []
    for i, xi in enumerate(x):
        lo, hi = bounds[i]
        step = max(abs_step, rel_step * max(1.0, abs(xi)))
        xp = list(x)
        xm = list(x)
        xp[i] = min(hi, xi + step)
        xm[i] = max(lo, xi - step)
        if xp[i] == xm[i]:
            g.append(0.0)
            continue
        fp = f(xp)
        fm = f(xm)
        g.append((fp - fm) / (xp[i] - xm[i]))
    return g


def topk(items, k: int):
    return sorted(items, key=lambda t: abs(t[1]), reverse=True)[:k]


def main() -> int:
    ap = argparse.ArgumentParser()
    ap.add_argument("--workspace", type=Path, required=True)
    ap.add_argument("--measurement", type=str, required=True)
    ap.add_argument("--out", type=Path, default=Path("tmp/pyhf_vs_nextstat_diff.json"))
    ap.add_argument("--top-bins", type=int, default=10)
    ap.add_argument("--top-params", type=int, default=12)
    ap.add_argument("--n-random", type=int, default=0, help="Also evaluate a few random points (smoke).")
    args = ap.parse_args()

    ws = json.loads(args.workspace.read_text())
    pyhf_model, pyhf_data = pyhf_model_and_data(ws, args.measurement)
    pyhf_init = list(map(float, pyhf_model.config.suggested_init()))
    pyhf_bounds = [(float(a), float(b)) for a, b in pyhf_model.config.suggested_bounds()]

    ns_model = nextstat.HistFactoryModel.from_workspace(json.dumps(ws))
    ns_names = ns_model.parameter_names()
    ns_init = ns_model.suggested_init()

    # sanity: same name set
    if set(ns_names) != set(pyhf_model.config.par_names):
        missing = sorted(set(pyhf_model.config.par_names) - set(ns_names))
        extra = sorted(set(ns_names) - set(pyhf_model.config.par_names))
        raise SystemExit(f"Parameter name sets differ. missing={missing} extra={extra}")

    def ns_params(pyhf_params: List[float]) -> List[float]:
        return map_params_by_name(pyhf_model.config.par_names, pyhf_params, ns_names, ns_init)

    def delta_nll(pyhf_params: List[float]) -> float:
        return float(ns_model.nll(ns_params(pyhf_params))) - pyhf_nll(pyhf_model, pyhf_data, pyhf_params)

    t0 = time.perf_counter()
    nll_py = pyhf_nll(pyhf_model, pyhf_data, pyhf_init)
    nll_ns = float(ns_model.nll(ns_params(pyhf_init)))
    dt_eval = time.perf_counter() - t0

    exp_py = [float(x) for x in pyhf_model.expected_data(pyhf_init)]
    exp_ns = [float(x) for x in ns_model.expected_data(ns_params(pyhf_init))]
    if len(exp_py) != len(exp_ns):
        raise SystemExit(f"expected_data length mismatch: pyhf={len(exp_py)} ns={len(exp_ns)}")

    exp_diff = [a - b for a, b in zip(exp_ns, exp_py)]
    bin_rank = topk(list(enumerate(exp_diff)), args.top_bins)

    sens = finite_diff_sensitivity(delta_nll, pyhf_init, pyhf_bounds)
    param_rank = topk(list(zip(pyhf_model.config.par_names, sens)), args.top_params)

    artifact: Dict[str, Any] = {
        "meta": {
            "workspace": str(args.workspace),
            "measurement": args.measurement,
            "pyhf_version": pyhf.__version__,
            "nextstat_version": nextstat.__version__,
            "timestamp": int(time.time()),
        },
        "nll": {
            "pyhf": nll_py,
            "nextstat": nll_ns,
            "delta": nll_ns - nll_py,
            "eval_wall_s": dt_eval,
        },
        "parameters": {
            "pyhf_order": list(pyhf_model.config.par_names),
            "pyhf_init": pyhf_init,
            "pyhf_bounds": pyhf_bounds,
            "nextstat_order": ns_names,
        },
        "expected_data": {
            "pyhf": exp_py,
            "nextstat": exp_ns,
            "delta": exp_diff,
            "max_abs_delta": max(abs(x) for x in exp_diff) if exp_diff else 0.0,
            "top_bins_by_abs_delta": [{"bin": int(i), "delta": float(d)} for i, d in bin_rank],
        },
        "sensitivity": {
            "notes": "finite differences of delta_nll wrt each pyhf-ordered parameter",
            "top_params_by_abs_grad": [{"name": n, "grad": float(g)} for n, g in param_rank],
        },
    }

    args.out.parent.mkdir(parents=True, exist_ok=True)
    args.out.write_text(json.dumps(artifact, indent=2))

    # concise stdout summary
    print(f"NLL pyhf={nll_py:.12g} nextstat={nll_ns:.12g} delta={nll_ns - nll_py:+.3e}")
    max_abs = artifact["expected_data"]["max_abs_delta"]
    print(f"expected_data max_abs_delta={max_abs:.3e}")
    if max_abs > 0:
        for row in artifact["expected_data"]["top_bins_by_abs_delta"]:
            print(f"  bin {row['bin']:>4}: delta={row['delta']:+.6g}")
    for row in artifact["sensitivity"]["top_params_by_abs_grad"]:
        g = row["grad"]
        if math.isfinite(g) and abs(g) > 0:
            print(f"  d(delta_nll)/d{row['name']} = {g:+.6g}")

    print(f"Wrote: {args.out}")
    return 0


if __name__ == "__main__":
    raise SystemExit(main())

