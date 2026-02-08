#!/usr/bin/env python3
"""Validate NextStat MLE against pyhf.

Compares best-fit parameters, uncertainties, and NLL values.
Parameters are aligned by NAME (not index) since pyhf and NextStat
may order them differently.
"""

import argparse
import json
import sys
import time
import numpy as np
try:
    import pyhf  # type: ignore
except ModuleNotFoundError as e:
    raise SystemExit(
        "Missing dependency: pyhf. Install it (e.g. `pip install pyhf`) to run tests/validate_mle.py."
    ) from e

import nextstat


def load_workspace(path: str):
    """Load tchannel workspace."""
    with open(path) as f:
        return json.load(f)


def pyhf_fit(workspace, *, compute_uncertainties: bool):
    """Fit with pyhf and (optionally) compute Hessian uncertainties."""
    t_build0 = time.perf_counter()
    ws = pyhf.Workspace(workspace)
    model = ws.model()
    data = ws.data(model)
    t_build = time.perf_counter() - t_build0

    t_fit0 = time.perf_counter()
    bestfit = pyhf.infer.mle.fit(data, model)
    t_fit = time.perf_counter() - t_fit0

    t_nll0 = time.perf_counter()
    nll = float(pyhf.infer.mle.twice_nll(bestfit, data, model)[0]) / 2.0
    t_nll = time.perf_counter() - t_nll0

    uncertainties = []
    t_unc = 0.0
    if compute_uncertainties:
        # Numerical Hessian for uncertainties.
        def nll_func(x: np.ndarray) -> float:
            return float(pyhf.infer.mle.twice_nll(x, data, model).item()) / 2.0

        x0 = np.asarray(bestfit, dtype=float)
        n = len(x0)
        hessian = np.zeros((n, n), dtype=float)
        t_unc0 = time.perf_counter()
        f0 = nll_func(x0)

        h_step = 1e-4
        damping = 1e-9

        for i in range(n):
            hi = h_step * max(abs(x0[i]), 1.0)
            xp = x0.copy(); xp[i] += hi
            xm = x0.copy(); xm[i] -= hi
            fp = nll_func(xp)
            fm = nll_func(xm)
            hessian[i, i] = (fp - 2.0 * f0 + fm) / (hi * hi)

            for j in range(i + 1, n):
                hj = h_step * max(abs(x0[j]), 1.0)
                xpp = x0.copy(); xpp[i] += hi; xpp[j] += hj
                xpm = x0.copy(); xpm[i] += hi; xpm[j] -= hj
                xmp = x0.copy(); xmp[i] -= hi; xmp[j] += hj
                xmm = x0.copy(); xmm[i] -= hi; xmm[j] -= hj
                fij = (nll_func(xpp) - nll_func(xpm) - nll_func(xmp) + nll_func(xmm)) / (4.0 * hi * hj)
                hessian[i, j] = fij
                hessian[j, i] = fij

        hessian = hessian + np.eye(n) * damping

        try:
            cov = np.linalg.inv(hessian)
        except np.linalg.LinAlgError:
            cov = np.linalg.pinv(hessian)

        uncertainties = np.sqrt(np.maximum(np.diag(cov), 0.0))
        t_unc = time.perf_counter() - t_unc0

    return {
        'names': list(model.config.par_names),
        'parameters': list(bestfit),
        'poi_name': model.config.par_names[model.config.poi_index],
        'uncertainties': list(uncertainties),
        'nll': nll,
        'timing_s': {
            'build': t_build,
            'fit': t_fit,
            'nll': t_nll,
            'uncertainties': t_unc,
            'total': t_build + t_fit + t_nll + t_unc,
        },
    }


def nextstat_fit(workspace):
    """Fit with NextStat."""
    t_ser0 = time.perf_counter()
    workspace_json = json.dumps(workspace)
    t_ser = time.perf_counter() - t_ser0

    t_build0 = time.perf_counter()
    model = nextstat.HistFactoryModel.from_workspace(workspace_json)
    t_build = time.perf_counter() - t_build0

    mle = nextstat.MaximumLikelihoodEstimator()
    t_fit0 = time.perf_counter()
    result = mle.fit(model)
    t_fit = time.perf_counter() - t_fit0

    return {
        'names': list(model.parameter_names()),
        'parameters': list(result.parameters),
        'uncertainties': list(result.uncertainties),
        'nll': result.nll,
        'converged': result.converged,
        'n_evaluations': result.n_evaluations,
        'timing_s': {
            'serialize_workspace': t_ser,
            'build': t_build,
            'fit': t_fit,
            'total': t_ser + t_build + t_fit,
        },
    }


def _build_name_map(pyhf_names, ns_names):
    """Build index mapping: for each pyhf index, the corresponding ns index."""
    ns_name_to_idx = {n: i for i, n in enumerate(ns_names)}
    pyhf_to_ns = {}
    for pi, name in enumerate(pyhf_names):
        ni = ns_name_to_idx.get(name)
        if ni is not None:
            pyhf_to_ns[pi] = ni
    return pyhf_to_ns


def compare_results(pyhf_res, nextstat_res):
    """Compare fit results, aligned by parameter name."""
    print("=" * 80)
    print("MLE VALIDATION: NextStat vs pyhf (name-aligned)")
    print("=" * 80)

    pyhf_names = pyhf_res['names']
    ns_names = nextstat_res['names']
    pyhf_to_ns = _build_name_map(pyhf_names, ns_names)

    pyhf_params = np.asarray(pyhf_res['parameters'], dtype=float)
    ns_params = np.asarray(nextstat_res['parameters'], dtype=float)
    pyhf_unc = np.asarray(pyhf_res.get('uncertainties') or [], dtype=float)
    ns_unc = np.asarray(nextstat_res.get('uncertainties') or [], dtype=float)

    n = len(pyhf_names)
    n_mapped = len(pyhf_to_ns)

    print(f"\n  pyhf params:    {len(pyhf_names)}")
    print(f"  NextStat params: {len(ns_names)}")
    print(f"  Name-matched:    {n_mapped}")
    if set(pyhf_names) != set(ns_names):
        only_pyhf = set(pyhf_names) - set(ns_names)
        only_ns = set(ns_names) - set(pyhf_names)
        if only_pyhf:
            print(f"  Only in pyhf:    {sorted(only_pyhf)}")
        if only_ns:
            print(f"  Only in NextStat: {sorted(only_ns)}")

    # --- POI ---
    poi_name = pyhf_res.get('poi_name', pyhf_names[0])
    ns_name_to_idx = {n: i for i, n in enumerate(ns_names)}
    pyhf_poi_idx = pyhf_names.index(poi_name)
    ns_poi_idx = ns_name_to_idx[poi_name]
    poi_pyhf = pyhf_params[pyhf_poi_idx]
    poi_ns = ns_params[ns_poi_idx]
    diff_poi = abs(poi_pyhf - poi_ns)

    print(f"\n  POI ({poi_name}):")
    print(f"    pyhf:     {poi_pyhf:+.6f}")
    print(f"    NextStat: {poi_ns:+.6f}")
    print(f"    Diff:     {diff_poi:.2e}")

    # --- Parameter comparison (name-aligned) ---
    PARAM_TOL = 5e-2
    UNC_RTOL = 0.5

    param_diffs = []
    rows_params = []

    for pi in range(n):
        ni = pyhf_to_ns.get(pi)
        if ni is None:
            continue
        name = pyhf_names[pi]
        p_val = float(pyhf_params[pi])
        n_val = float(ns_params[ni])
        d_param = abs(p_val - n_val)
        param_diffs.append(d_param)
        rows_params.append((name, p_val, n_val, d_param))

    param_diffs = np.array(param_diffs)

    n_param_ok = int((param_diffs < PARAM_TOL).sum())
    # Uncertainty diffs are computed only when pyhf uncertainties are available.
    has_pyhf_unc = pyhf_unc.size == n

    print(f"\n  PARAMETERS (tol={PARAM_TOL}):")
    print(f"    Match: {n_param_ok}/{n_mapped}")
    print(f"    Max diff:  {param_diffs.max():.4e}")
    print(f"    Mean diff: {param_diffs.mean():.4e}")

    # Show worst 15
    sorted_rows = sorted(rows_params, key=lambda r: -r[3])
    print(f"\n  {'Name':<45} {'pyhf':>10} {'NextStat':>10} {'Diff':>10}")
    print("  " + "-" * 77)
    for name, p_val, n_val, d_param in sorted_rows[:15]:
        tag = " *" if d_param >= PARAM_TOL else ""
        print(f"  {name:<45} {p_val:>+10.5f} {n_val:>+10.5f} {d_param:>10.4e}{tag}")

    # --- Uncertainty comparison ---
    if not has_pyhf_unc:
        print(f"\n  UNCERTAINTIES: skipped (pyhf uncertainties not computed)")
    else:
        # Build uncertainty diff rows (name-aligned) now that we know arrays exist.
        rows_unc = []
        for pi in range(n):
            ni = pyhf_to_ns.get(pi)
            if ni is None:
                continue
            name = pyhf_names[pi]
            p_unc = float(pyhf_unc[pi])
            n_unc = float(ns_unc[ni]) if ns_unc.size == len(ns_names) else float("nan")
            d_unc = abs(p_unc - n_unc) if np.isfinite(n_unc) else float("nan")
            rows_unc.append((name, p_unc, n_unc, d_unc))

        n_unc_ok = int(np.array([
            abs(r[1] - r[2]) < max(UNC_RTOL * max(abs(r[1]), abs(r[2])), 1e-3)
            for r in rows_unc
        ]).sum())
        print(f"\n  UNCERTAINTIES (rtol={UNC_RTOL}):")
        print(f"    Match: {n_unc_ok}/{n_mapped}")

        sorted_unc = sorted(rows_unc, key=lambda r: -float(r[3] if np.isfinite(r[3]) else -1.0))
        print(f"\n  {'Name':<45} {'pyhf':>10} {'NextStat':>10} {'Diff':>10}")
        print("  " + "-" * 77)
        for name, p_unc, n_unc, d_unc in sorted_unc[:15]:
            print(f"  {name:<45} {p_unc:>10.5f} {n_unc:>10.5f} {d_unc:>10.4e}")

    # --- NLL ---
    diff_nll = abs(pyhf_res['nll'] - nextstat_res['nll'])
    NLL_TOL = 1.0  # for large models, optimizer noise is expected
    nll_ok = diff_nll < NLL_TOL

    print(f"\n  NLL:")
    print(f"    pyhf:     {pyhf_res['nll']:.6f}")
    print(f"    NextStat: {nextstat_res['nll']:.6f}")
    print(f"    Diff:     {diff_nll:.4e} {'OK' if nll_ok else 'LARGE'}")

    # --- Convergence ---
    print(f"\n  CONVERGENCE:")
    print(f"    Converged:   {nextstat_res['converged']}")
    print(f"    Evaluations: {nextstat_res['n_evaluations']}")

    # --- Timing ---
    t_pyhf = pyhf_res.get("timing_s", {})
    t_ns = nextstat_res.get("timing_s", {})
    print(f"\n  TIMING (seconds):")
    print(f"    pyhf build:         {t_pyhf.get('build', float('nan')):.3f}")
    print(f"    pyhf fit:           {t_pyhf.get('fit', float('nan')):.3f}")
    print(f"    pyhf nll:           {t_pyhf.get('nll', float('nan')):.3f}")
    print(f"    pyhf uncertainties: {t_pyhf.get('uncertainties', float('nan')):.3f}")
    print(f"    pyhf total:         {t_pyhf.get('total', float('nan')):.3f}")
    print(f"    NextStat serialize: {t_ns.get('serialize_workspace', float('nan')):.3f}")
    print(f"    NextStat build:     {t_ns.get('build', float('nan')):.3f}")
    print(f"    NextStat fit:       {t_ns.get('fit', float('nan')):.3f}")
    print(f"    NextStat total:     {t_ns.get('total', float('nan')):.3f}")
    speedup = t_pyhf.get('total', 0) / max(t_ns.get('total', 1e-9), 1e-9)
    print(f"    Speedup:            {speedup:.1f}x")

    # --- Verdict ---
    # For a 277-param model, we use relaxed tolerances:
    # params within 5e-2 for >95% of parameters, NLL within 1.0
    param_pass_rate = n_param_ok / max(n_mapped, 1)
    passed = param_pass_rate >= 0.95 and nll_ok and diff_poi < 0.05

    print("\n" + "=" * 80)
    if passed:
        print(f"PASSED: POI diff={diff_poi:.4e}, param match={param_pass_rate:.1%}, NLL diff={diff_nll:.4e}")
    else:
        reasons = []
        if diff_poi >= 0.05:
            reasons.append(f"POI diff={diff_poi:.4e}")
        if param_pass_rate < 0.95:
            reasons.append(f"param match={param_pass_rate:.1%} < 95%")
        if not nll_ok:
            reasons.append(f"NLL diff={diff_nll:.4e}")
        print(f"FAILED: {', '.join(reasons)}")

    return 0 if passed else 1


def main():
    """Main validation."""
    ap = argparse.ArgumentParser()
    ap.add_argument(
        "--workspace",
        default="tests/fixtures/tchannel_workspace.json",
        help="Path to pyhf workspace.json fixture",
    )
    ap.add_argument(
        "--skip-uncertainties",
        action="store_true",
        help="Skip expensive numerical Hessian uncertainties for pyhf (timing/fit parity only).",
    )
    args = ap.parse_args()

    t_all0 = time.perf_counter()
    print("Loading workspace...")
    workspace = load_workspace(str(args.workspace))

    print("Fitting with pyhf...")
    pyhf_res = pyhf_fit(workspace, compute_uncertainties=not bool(args.skip_uncertainties))

    print("Fitting with NextStat...")
    nextstat_res = nextstat_fit(workspace)

    rc = compare_results(pyhf_res, nextstat_res)
    t_all = time.perf_counter() - t_all0
    print(f"\nTOTAL wall time: {t_all:.3f} s")
    return rc


if __name__ == "__main__":
    sys.exit(main())
