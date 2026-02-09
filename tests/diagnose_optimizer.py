#!/usr/bin/env python3
"""Optimizer diagnostic: cross-init, multi-start, gradient norms.

This script answers the key question:
  "When NextStat and pyhf find different optima, is the objective different
   or is it just optimizer convergence quality?"

Diagnostics per workspace:
  1. Gradient norms (raw + projected/KKT) at both optima
  2. Cross-init: pyhf started from NS hat, NS started from pyhf hat
  3. Multi-start pyhf: N random inits within bounds (pre-validated)
  4. Verdict: optimizer quality vs model mismatch

Usage:
  PYTHONPATH=bindings/ns-py/python .venv/bin/python tests/diagnose_optimizer.py
  PYTHONPATH=bindings/ns-py/python .venv/bin/python tests/diagnose_optimizer.py \
      --workspace tests/fixtures/workspace_tHu.json \
      --workspace tests/fixtures/tttt-prod_workspace.json \
      --multi-start 20
"""

from __future__ import annotations

import argparse
import json
import math
import os
import statistics
import sys
import time
import traceback
from collections import Counter
from pathlib import Path
from typing import Any, Dict, List, Optional, Tuple

import numpy as np


# ---------------------------------------------------------------------------
# Helpers (shared patterns from repeat_mle_fits.py)
# ---------------------------------------------------------------------------

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


def _measurement_name(ws: dict) -> str:
    ms = ws.get("measurements", [])
    if not ms:
        raise ValueError("workspace has no measurements")
    name = ms[0].get("name")
    if not isinstance(name, str) or not name:
        raise ValueError("measurement has no name")
    return name


def _remap_params(
    src_names: List[str], src_params: List[float], dst_names: List[str],
) -> List[float]:
    dst_index = {n: i for i, n in enumerate(dst_names)}
    out = [0.0] * len(dst_names)
    for name, value in zip(src_names, src_params):
        if name not in dst_index:
            raise KeyError(f"missing parameter '{name}' in destination")
        out[dst_index[name]] = float(value)
    return out


def _map_params_by_name(
    src_names: List[str], src_params: List[float],
    dst_names: List[str], dst_init: List[float],
) -> List[float]:
    dst_index = {name: i for i, name in enumerate(dst_names)}
    out = list(dst_init)
    for name, value in zip(src_names, src_params):
        if name in dst_index:
            out[dst_index[name]] = float(value)
    return out


def _iter_default_workspace_paths():
    fixtures = Path("tests/fixtures")
    if fixtures.exists():
        for p in sorted(fixtures.glob("*.json")):
            yield p


# ---------------------------------------------------------------------------
# Diagnostics
# ---------------------------------------------------------------------------

def grad_norm(model: Any, params: List[float]) -> Optional[float]:
    """L2 norm of gradient (via NextStat model.grad_nll)."""
    try:
        g = model.grad_nll(params)
        return float(math.sqrt(sum(x * x for x in g)))
    except Exception:
        return None


def _projected_grad_norm_impl(
    grad: List[float], params: List[float], bounds: List[Tuple[float, float]],
    tol: float = 1e-12,
) -> float:
    """Projected gradient norm for bounded optimisation (KKT residual).

    At a bound-constrained minimum the *projected* gradient is zero:
      - if param_i ~ lo_i  and  grad_i > 0  -> projected component = 0
      - if param_i ~ hi_i  and  grad_i < 0  -> projected component = 0
      - otherwise                            -> projected component = grad_i
    """
    pg = []
    for g_i, p_i, (lo, hi) in zip(grad, params, bounds):
        if p_i <= lo + tol and g_i > 0:
            pg.append(0.0)
        elif p_i >= hi - tol and g_i < 0:
            pg.append(0.0)
        else:
            pg.append(float(g_i))
    return float(math.sqrt(sum(x * x for x in pg)))


def _n_at_bounds_impl(
    params: List[float], bounds: List[Tuple[float, float]], tol: float = 1e-12,
) -> int:
    """Count how many parameters sit at their bounds."""
    count = 0
    for p_i, (lo, hi) in zip(params, bounds):
        if p_i <= lo + tol or p_i >= hi - tol:
            count += 1
    return count


def grad_diagnostics(
    model: Any, params: List[float], bounds: List[Tuple[float, float]],
) -> Dict[str, Any]:
    """Return raw grad norm, projected grad norm, and n_at_bounds."""
    out: Dict[str, Any] = {}
    try:
        g = model.grad_nll(params)
        out["grad_norm"] = float(math.sqrt(sum(x * x for x in g)))
        out["proj_grad_norm"] = _projected_grad_norm_impl(g, params, bounds)
        out["n_at_bounds"] = _n_at_bounds_impl(params, bounds)
    except Exception:
        out["grad_norm"] = None
        out["proj_grad_norm"] = None
        out["n_at_bounds"] = None
    return out


def fit_pyhf(pyhf, model, data, fit_options, init_pars=None):
    """Fit with pyhf, return (nll, params, wall_s) or raise."""
    opts = dict(fit_options)
    if init_pars is not None:
        opts["init_pars"] = init_pars
    t0 = time.perf_counter()
    bestfit, twice_nll = pyhf.infer.mle.fit(data, model, return_fitted_val=True, **opts)
    wall_s = time.perf_counter() - t0
    nll = float(twice_nll.item()) / 2.0
    params = [float(x) for x in list(bestfit)]
    return nll, params, wall_s


def fit_nextstat(mle, model, init_pars=None, data=None):
    """Fit with NextStat, return (nll, params, wall_s, converged) or raise."""
    t0 = time.perf_counter()
    if init_pars is not None:
        res = mle.fit(model, init_pars=init_pars, data=data)
    elif data is not None:
        res = mle.fit(model, data=data)
    else:
        res = mle.fit(model)
    wall_s = time.perf_counter() - t0
    return float(res.nll), [float(x) for x in list(res.parameters)], wall_s, bool(getattr(res, "converged", True))


def diagnose_workspace(
    *,
    pyhf,
    nextstat,
    ws_dict: dict,
    meas_name: str,
    pyhf_fit_options: dict,
    ns_mle_config: dict,
    n_multi_start: int,
    multi_start_seed: int,
) -> dict:
    """Run full diagnostic for one workspace."""
    result: Dict[str, Any] = {}

    # Build models
    py_ws = pyhf.Workspace(ws_dict)
    py_model = py_ws.model(
        measurement_name=meas_name,
        modifier_settings={
            "normsys": {"interpcode": "code4"},
            "histosys": {"interpcode": "code4p"},
        },
    )
    py_data = py_ws.data(py_model)
    py_names = list(py_model.config.par_names)

    ns_model = nextstat.HistFactoryModel.from_workspace(json.dumps(ws_dict))
    ns_names = list(ns_model.parameter_names())
    mle = nextstat.MaximumLikelihoodEstimator(**ns_mle_config)

    if set(py_names) != set(ns_names):
        result["error"] = "parameter sets differ"
        return result

    result["n_params"] = len(py_names)

    # --- Step 1: Default fits ---
    print("  [1/4] Default fits ...", flush=True)
    try:
        py_nll, py_params, py_s = fit_pyhf(pyhf, py_model, py_data, pyhf_fit_options)
        result["pyhf_default"] = {"nll": py_nll, "wall_s": py_s}
    except Exception:
        result["pyhf_default"] = {"error": traceback.format_exc()[:300]}
        return result

    try:
        ns_nll, ns_params, ns_s, ns_conv = fit_nextstat(mle, ns_model)
        result["ns_default"] = {"nll": ns_nll, "wall_s": ns_s, "converged": ns_conv}
    except Exception:
        result["ns_default"] = {"error": traceback.format_exc()[:300]}
        return result

    result["nll_delta_ns_minus_pyhf"] = ns_nll - py_nll

    # --- Step 2: Gradient norms (raw + projected KKT) ---
    print("  [2/4] Gradient norms ...", flush=True)
    ns_bounds = list(ns_model.suggested_bounds())

    ns_gd = grad_diagnostics(ns_model, ns_params, ns_bounds)
    result["grad_norm_at_ns_hat"] = ns_gd["grad_norm"]
    result["proj_grad_norm_at_ns_hat"] = ns_gd["proj_grad_norm"]
    result["n_at_bounds_ns_hat"] = ns_gd["n_at_bounds"]

    py_in_ns = _remap_params(py_names, py_params, ns_names)
    py_gd = grad_diagnostics(ns_model, py_in_ns, ns_bounds)
    result["grad_norm_at_pyhf_hat"] = py_gd["grad_norm"]
    result["proj_grad_norm_at_pyhf_hat"] = py_gd["proj_grad_norm"]
    result["n_at_bounds_pyhf_hat"] = py_gd["n_at_bounds"]

    # --- Step 3: Cross-init fits ---
    print("  [3/4] Cross-init fits ...", flush=True)

    # pyhf from NS hat
    try:
        ns_hat_in_py = _map_params_by_name(
            ns_names, ns_params,
            py_names, list(py_model.config.suggested_init()),
        )
        ci_py_nll, ci_py_params, ci_py_s = fit_pyhf(
            pyhf, py_model, py_data, pyhf_fit_options, init_pars=ns_hat_in_py,
        )
        result["pyhf_from_ns_hat"] = {
            "nll": ci_py_nll,
            "wall_s": ci_py_s,
            "nll_improvement": py_nll - ci_py_nll,
        }
    except Exception:
        result["pyhf_from_ns_hat"] = {"error": traceback.format_exc()[:300]}

    # NS from pyhf hat
    try:
        ci_ns_nll, ci_ns_params, ci_ns_s, ci_ns_conv = fit_nextstat(
            mle, ns_model, init_pars=py_in_ns,
        )
        result["ns_from_pyhf_hat"] = {
            "nll": ci_ns_nll,
            "wall_s": ci_ns_s,
            "converged": ci_ns_conv,
            "nll_improvement": ns_nll - ci_ns_nll,
        }
    except Exception:
        result["ns_from_pyhf_hat"] = {"error": traceback.format_exc()[:300]}

    # --- Step 4: Multi-start pyhf ---
    if n_multi_start > 0:
        print(f"  [4/4] Multi-start pyhf ({n_multi_start} starts) ...", flush=True)
        rng = np.random.default_rng(multi_start_seed)
        pyhf_init = list(py_model.config.suggested_init())
        pyhf_bounds = list(py_model.config.suggested_bounds())
        n_pars = len(pyhf_init)

        ms_nlls: List[float] = []
        ms_fails = 0
        ms_invalid_init = 0
        fail_reasons: Counter = Counter()

        max_attempts = n_multi_start * 5  # extra budget for invalid inits
        attempts = 0

        while len(ms_nlls) + ms_fails < n_multi_start and attempts < max_attempts:
            attempts += 1
            init = []
            for k in range(n_pars):
                lo, hi = pyhf_bounds[k]
                center = pyhf_init[k]
                sigma = 0.1 * max(hi - lo, 0.01)
                val = float(rng.normal(center, sigma))
                val = max(lo + 1e-6, min(hi - 1e-6, val))
                init.append(val)

            # Pre-validate: NLL must be finite at init point
            try:
                init_tensor = pyhf.tensorlib.astensor(init)
                data_tensor = pyhf.tensorlib.astensor(py_data)
                logpdf_val = py_model.logpdf(init_tensor, data_tensor)
                nll_check = -float(pyhf.tensorlib.tolist(logpdf_val)[0])
                if not math.isfinite(nll_check):
                    ms_invalid_init += 1
                    continue
            except Exception:
                ms_invalid_init += 1
                continue

            try:
                ms_nll, _, _ = fit_pyhf(pyhf, py_model, py_data, pyhf_fit_options, init_pars=init)
                ms_nlls.append(ms_nll)
            except Exception as exc:
                ms_fails += 1
                reason = str(exc).split("\n")[0][:120]
                fail_reasons[reason] += 1

        best_ms = min(ms_nlls) if ms_nlls else float("nan")
        result["pyhf_multi_start"] = {
            "n_starts": len(ms_nlls) + ms_fails,
            "n_ok": len(ms_nlls),
            "n_failed": ms_fails,
            "n_invalid_init_skipped": ms_invalid_init,
            "n_attempts": attempts,
            "best_nll": float(best_ms),
            "worst_nll": float(max(ms_nlls)) if ms_nlls else float("nan"),
            "mean_nll": float(statistics.mean(ms_nlls)) if ms_nlls else float("nan"),
            "stdev_nll": float(statistics.pstdev(ms_nlls)) if len(ms_nlls) >= 2 else 0.0,
            "default_nll": py_nll,
            "best_improvement_over_default": float(py_nll - best_ms) if math.isfinite(best_ms) else float("nan"),
            "top_failure_reasons": [
                {"reason": reason, "count": count}
                for reason, count in fail_reasons.most_common(3)
            ],
        }
    else:
        print("  [4/4] Multi-start skipped (n=0)", flush=True)

    # --- Verdict ---
    ci_py_nll_val = result.get("pyhf_from_ns_hat", {}).get("nll")
    ms_best = result.get("pyhf_multi_start", {}).get("best_nll")
    best_pyhf_achievable = py_nll
    if ci_py_nll_val is not None and math.isfinite(ci_py_nll_val):
        best_pyhf_achievable = min(best_pyhf_achievable, ci_py_nll_val)
    if ms_best is not None and math.isfinite(ms_best):
        best_pyhf_achievable = min(best_pyhf_achievable, ms_best)

    gap = best_pyhf_achievable - ns_nll
    if abs(gap) < 1e-4:
        verdict = "MATCH: pyhf can reach NS level with better init -> optimizer init issue in pyhf"
    elif gap > 0:
        verdict = f"NS_BETTER: NS still {gap:.6e} lower than best pyhf achievable -> NS optimizer genuinely superior"
    else:
        verdict = f"PYHF_BETTER: pyhf achievable is {-gap:.6e} lower than NS -> investigate NS convergence"

    result["verdict"] = verdict
    result["best_pyhf_achievable"] = best_pyhf_achievable

    return result


# ---------------------------------------------------------------------------
# Pretty print
# ---------------------------------------------------------------------------

def print_result(ws_path: str, r: dict) -> None:
    print(f"\n{'='*70}")
    print(f"  {ws_path}  ({r.get('n_params', '?')} params)")
    print(f"{'='*70}")

    if r.get("error"):
        print(f"  ERROR: {r['error']}")
        return

    py_d = r.get("pyhf_default", {})
    ns_d = r.get("ns_default", {})
    print(f"  Default pyhf NLL:     {py_d.get('nll', '?'):.10g}  ({py_d.get('wall_s', 0):.4f}s)")
    print(f"  Default NextStat NLL: {ns_d.get('nll', '?'):.10g}  ({ns_d.get('wall_s', 0):.4f}s)")
    delta = r.get("nll_delta_ns_minus_pyhf", float("nan"))
    print(f"  Delta (NS - pyhf):    {delta:+.6e}")

    # Gradient norms
    gn1 = r.get("grad_norm_at_ns_hat")
    pgn1 = r.get("proj_grad_norm_at_ns_hat")
    nab1 = r.get("n_at_bounds_ns_hat")
    gn2 = r.get("grad_norm_at_pyhf_hat")
    pgn2 = r.get("proj_grad_norm_at_pyhf_hat")
    nab2 = r.get("n_at_bounds_pyhf_hat")

    if gn1 is not None:
        extra = ""
        if pgn1 is not None:
            extra += f"  proj={pgn1:.4e}"
        if nab1 is not None:
            extra += f"  at_bounds={nab1}/{r.get('n_params', '?')}"
        print(f"  ||grad||@NS_hat:      {gn1:.4e}{extra}")

    if gn2 is not None:
        extra = ""
        if pgn2 is not None:
            extra += f"  proj={pgn2:.4e}"
        if nab2 is not None:
            extra += f"  at_bounds={nab2}/{r.get('n_params', '?')}"
        print(f"  ||grad||@pyhf_hat:    {gn2:.4e}{extra}")

    ci_py = r.get("pyhf_from_ns_hat", {})
    if ci_py.get("nll") is not None:
        imp = ci_py.get("nll_improvement", 0)
        print(f"  pyhf(init=NS_hat):    NLL={ci_py['nll']:.10g}  improvement={imp:+.4e}")

    ci_ns = r.get("ns_from_pyhf_hat", {})
    if ci_ns.get("nll") is not None:
        imp = ci_ns.get("nll_improvement", 0)
        print(f"  NS(init=pyhf_hat):    NLL={ci_ns['nll']:.10g}  improvement={imp:+.4e}")

    ms = r.get("pyhf_multi_start", {})
    if ms.get("n_starts"):
        n_inv = ms.get("n_invalid_init_skipped", 0)
        print(
            f"  pyhf multi-start ({ms['n_starts']}): "
            f"ok={ms['n_ok']} fail={ms['n_failed']} invalid_init={n_inv}  "
            f"best={ms['best_nll']:.10g}  "
            f"improve_vs_default={ms['best_improvement_over_default']:+.4e}"
        )
        for fr in ms.get("top_failure_reasons", []):
            print(f"    fail({fr['count']}x): {fr['reason']}")

    print(f"\n  >>> VERDICT: {r.get('verdict', '?')}")
    print()


# ---------------------------------------------------------------------------
# Main
# ---------------------------------------------------------------------------

def main(argv=None) -> int:
    ap = argparse.ArgumentParser(description="Optimizer diagnostic: cross-init + multi-start")
    ap.add_argument("--workspace", action="append", default=[], help="Workspace JSON path (repeatable)")
    ap.add_argument("--exclude", action="append", default=[], help="Exclude workspace if substring matches path")
    ap.add_argument("--multi-start", type=int, default=10, help="Number of pyhf multi-start fits per workspace")
    ap.add_argument("--seed", type=int, default=42)
    ap.add_argument("--threads", type=int, default=1)
    ap.add_argument("--out-json", type=Path, default=Path("tmp/diagnose_optimizer.json"))
    ap.add_argument("--pyhf-method", default="SLSQP")
    ap.add_argument("--pyhf-maxiter", type=int, default=100000)
    ap.add_argument("--pyhf-tolerance", type=float, default=float("nan"))
    ap.add_argument("--pyhf-do-grad", type=int, default=0,
                    help="0=force do_grad off (explicit numpy backend), 1=let pyhf decide")
    ap.add_argument("--ns-max-iter", type=int, default=1000)
    ap.add_argument("--ns-tol", type=float, default=1e-6)
    ap.add_argument("--ns-m", type=int, default=10)
    args = ap.parse_args(argv)

    threads = int(args.threads)
    if threads >= 1:
        for k in [
            "RAYON_NUM_THREADS", "OMP_NUM_THREADS", "MKL_NUM_THREADS",
            "OPENBLAS_NUM_THREADS", "VECLIB_MAXIMUM_THREADS", "NUMEXPR_NUM_THREADS",
        ]:
            os.environ[k] = str(threads)

    import nextstat
    try:
        import pyhf
    except ImportError:
        raise SystemExit("pyhf required: pip install pyhf")

    # Force numpy backend with explicit do_grad control
    if args.pyhf_do_grad == 0:
        pyhf.set_backend("numpy", pyhf.optimize.scipy_optimizer(solver_options={"do_grad": False}))
        print("[config] pyhf backend=numpy, do_grad=False", flush=True)
    else:
        print(f"[config] pyhf backend={pyhf.tensorlib.name}, do_grad=pyhf default", flush=True)

    pyhf_fit_options: dict = {"method": args.pyhf_method, "maxiter": args.pyhf_maxiter}
    if math.isfinite(args.pyhf_tolerance):
        pyhf_fit_options["tolerance"] = args.pyhf_tolerance

    ns_mle_config = {"max_iter": args.ns_max_iter, "tol": args.ns_tol, "m": args.ns_m}

    paths = [Path(p) for p in args.workspace] if args.workspace else list(_iter_default_workspace_paths())
    excludes = list(args.exclude)
    if not args.workspace:
        excludes.append("tchannel_workspace.json")

    all_results: List[dict] = []

    for p in paths:
        if any(ex in str(p) for ex in excludes):
            continue

        try:
            ws = json.loads(p.read_text())
        except Exception:
            print(f"[skip] {p}: cannot load JSON")
            continue

        if not _is_pyhf_workspace(ws):
            continue

        try:
            meas = _measurement_name(ws)
        except Exception:
            continue

        print(f"\n[diagnose] {p}", flush=True)

        try:
            r = diagnose_workspace(
                pyhf=pyhf,
                nextstat=nextstat,
                ws_dict=ws,
                meas_name=meas,
                pyhf_fit_options=pyhf_fit_options,
                ns_mle_config=ns_mle_config,
                n_multi_start=args.multi_start,
                multi_start_seed=args.seed,
            )
        except Exception:
            r = {"error": traceback.format_exc()[:500]}

        r["path"] = str(p)
        all_results.append(r)
        print_result(str(p), r)

    # Write JSON
    args.out_json.parent.mkdir(parents=True, exist_ok=True)
    args.out_json.write_text(json.dumps(all_results, indent=2, sort_keys=True) + "\n")
    print(f"\nWrote: {args.out_json}")

    return 0


if __name__ == "__main__":
    raise SystemExit(main())
