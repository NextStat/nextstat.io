#!/usr/bin/env python3
"""Minimal HEP benchmark seed: NLL parity + timing (pyhf vs NextStat).

This is intentionally small and self-contained so outsiders can rerun it.
"""

from __future__ import annotations

import argparse
import hashlib
import json
import platform
import sys
import timeit
from array import array
from pathlib import Path
from typing import Any, Callable

import pyhf

import nextstat


NLL_SANITY_ATOL = 1e-8
NLL_SANITY_RTOL = 1e-12


def sha256_file(path: Path) -> str:
    h = hashlib.sha256()
    with path.open("rb") as f:
        for chunk in iter(lambda: f.read(1024 * 1024), b""):
            h.update(chunk)
    return h.hexdigest()


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


def bench_time_per_call(fn: Callable[[], Any], *, target_s: float = 0.25, repeat: int = 5) -> float:
    number = 1
    while True:
        t = timeit.timeit(fn, number=number)
        if t >= target_s or number >= 1_000_000:
            break
        number *= 2
    times = [timeit.timeit(fn, number=number) / number for _ in range(repeat)]
    return min(times)


def main() -> int:
    parser = argparse.ArgumentParser()
    parser.add_argument(
        "--workspace",
        default=str(Path(__file__).resolve().parent / "datasets/simple_workspace.json"),
        help="Path to a pyhf workspace JSON.",
    )
    parser.add_argument(
        "--measurement-name",
        default="GaussExample",
        help="pyhf measurement name to load.",
    )
    parser.add_argument("--out", required=True, help="Output JSON path.")
    parser.add_argument("--deterministic", action="store_true", help="Deterministic output.")
    args = parser.parse_args()

    ws_path = Path(args.workspace).resolve()
    out_path = Path(args.out).resolve()
    out_path.parent.mkdir(parents=True, exist_ok=True)

    workspace: dict[str, Any] = json.loads(ws_path.read_text())
    pyhf_model, pyhf_data = pyhf_model_and_data(workspace, args.measurement_name)
    pyhf_params = list(pyhf_model.config.suggested_init())

    ns_model = nextstat.HistFactoryModel.from_workspace(json.dumps(workspace))
    ns_params_list = map_params_by_name(
        pyhf_model.config.par_names,
        pyhf_params,
        ns_model.parameter_names(),
        ns_model.suggested_init(),
    )
    # Buffer-protocol input to avoid per-element Python overhead for large vectors.
    ns_params = array("d", ns_params_list)

    pyhf_val = pyhf_nll(pyhf_model, pyhf_data, pyhf_params)
    ns_val = float(ns_model.nll(ns_params))
    abs_diff = abs(ns_val - pyhf_val)
    rel_diff = abs_diff / max(abs(ns_val), abs(pyhf_val), 1.0)
    ok = not (abs_diff > NLL_SANITY_ATOL and rel_diff > NLL_SANITY_RTOL)

    def f_pyhf():
        return pyhf_nll(pyhf_model, pyhf_data, pyhf_params)

    def f_ns():
        return ns_model.nll(ns_params)

    # Warmup (avoid one-time allocations dominating timing).
    for _ in range(10):
        f_pyhf()
        f_ns()

    pyhf_t = bench_time_per_call(f_pyhf)
    ns_t = bench_time_per_call(f_ns)
    speedup = pyhf_t / ns_t if ns_t > 0 else float("inf")

    n_main_bins = sum(len(obs["data"]) for obs in workspace.get("observations", []))

    doc: dict[str, Any] = {
        "schema_version": "nextstat.benchmark_result.v1",
        "suite": "hep",
        "case": "simple_workspace_nll",
        "deterministic": bool(args.deterministic),
        "meta": {
            "python": sys.version.split()[0],
            "platform": platform.platform(),
            "pyhf_version": pyhf.__version__,
            "nextstat_version": nextstat.__version__,
        },
        "dataset": {
            "path": str(ws_path),
            "sha256": sha256_file(ws_path),
        },
        "model": {"n_main_bins": int(n_main_bins), "n_params": int(ns_model.n_params())},
        "parity": {
            "ok": bool(ok),
            "nll_pyhf": float(pyhf_val),
            "nll_nextstat": float(ns_val),
            "abs_diff": float(abs_diff),
            "rel_diff": float(rel_diff),
            "atol": float(NLL_SANITY_ATOL),
            "rtol": float(NLL_SANITY_RTOL),
        },
        "timing": {
            "nll_time_s_per_call": {"pyhf": float(pyhf_t), "nextstat": float(ns_t)},
            "speedup_pyhf_over_nextstat": float(speedup),
        },
    }

    if not ok:
        # Write the artifact anyway (for debugging) but return non-zero.
        out_path.write_text(json.dumps(doc, indent=2, sort_keys=True) + "\n")
        return 2

    out_path.write_text(json.dumps(doc, indent=2, sort_keys=True) + "\n")
    return 0


if __name__ == "__main__":
    raise SystemExit(main())

