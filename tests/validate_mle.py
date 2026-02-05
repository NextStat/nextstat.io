#!/usr/bin/env python3
"""Validate NextStat MLE against pyhf."""

import json
import sys
import numpy as np
import pyhf

import nextstat

def load_workspace():
    """Load simple workspace."""
    with open("tests/fixtures/simple_workspace.json") as f:
        return json.load(f)

def pyhf_fit(workspace):
    """Fit with pyhf."""
    model = pyhf.Workspace(workspace).model()
    data = pyhf.Workspace(workspace).data(model)

    # Fit returns bestfit parameters
    bestfit = pyhf.infer.mle.fit(data, model)

    # Compute NLL at best-fit point
    nll = float(pyhf.infer.mle.twice_nll(bestfit, data, model)[0]) / 2.0

    # Numerical Hessian for uncertainties (pyhf doesn't expose it directly)
    def nll_func(x: np.ndarray) -> float:
        return float(pyhf.infer.mle.twice_nll(x, data, model).item()) / 2.0

    x0 = np.asarray(bestfit, dtype=float)
    n = len(x0)
    hessian = np.zeros((n, n), dtype=float)
    f0 = nll_func(x0)

    # Match NextStat defaults (see crates/ns-inference/src/mle.rs)
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

    return {
        'parameters': list(bestfit),
        'uncertainties': uncertainties,
        'nll': nll
    }

def nextstat_fit(workspace):
    """Fit with NextStat."""
    workspace_json = json.dumps(workspace)
    model = nextstat.HistFactoryModel.from_workspace(workspace_json)

    mle = nextstat.MaximumLikelihoodEstimator()
    result = mle.fit(model)

    return {
        'parameters': result.parameters,
        'uncertainties': result.uncertainties,
        'nll': result.nll,
        'converged': result.converged,
        'n_evaluations': result.n_evaluations
    }

def compare_results(pyhf_res, nextstat_res):
    """Compare fit results."""
    print("=" * 60)
    print("MLE VALIDATION: NextStat vs pyhf")
    print("=" * 60)

    print("\n📊 PARAMETERS:")
    print(f"{'Parameter':<15} {'pyhf':<15} {'NextStat':<15} {'Diff':<15}")
    print("-" * 60)

    params_match = True
    for i, (p_pyhf, p_ns) in enumerate(zip(pyhf_res['parameters'], nextstat_res['parameters'])):
        diff = abs(p_pyhf - p_ns)
        # Relaxed tolerance: optimizers can differ slightly while finding same minimum
        status = "✅" if diff < 2e-4 else "❌"
        print(f"param[{i}] {status}    {p_pyhf:<15.6f} {p_ns:<15.6f} {diff:<15.2e}")
        if diff >= 2e-4:
            params_match = False

    print("\n📏 UNCERTAINTIES:")
    print(f"{'Parameter':<15} {'pyhf':<15} {'NextStat':<15} {'Diff':<15}")
    print("-" * 60)

    unc_match = True
    for i, (u_pyhf, u_ns) in enumerate(zip(pyhf_res['uncertainties'], nextstat_res['uncertainties'])):
        diff = abs(u_pyhf - u_ns)
        status = "✅" if diff < 5e-4 else "❌"
        print(f"param[{i}] {status}    {u_pyhf:<15.6f} {u_ns:<15.6f} {diff:<15.2e}")
        if diff >= 5e-4:
            unc_match = False

    print("\n🎯 NEGATIVE LOG-LIKELIHOOD:")
    print(f"  pyhf:     {pyhf_res['nll']:.6f}")
    print(f"  NextStat: {nextstat_res['nll']:.6f}")
    diff_nll = abs(pyhf_res['nll'] - nextstat_res['nll'])
    status_nll = "✅" if diff_nll < 1e-4 else "❌"
    print(f"  Diff:     {diff_nll:.2e} {status_nll}")

    nll_match = diff_nll < 1e-4

    print("\n⚙️  CONVERGENCE:")
    print(f"  Converged: {nextstat_res['converged']}")
    print(f"  Evaluations: {nextstat_res['n_evaluations']}")

    print("\n" + "=" * 60)
    if params_match and nll_match and unc_match:
        print("✅ VALIDATION PASSED: Parameters, uncertainties, and NLL match within tolerance")
        return 0
    else:
        print("❌ VALIDATION FAILED: Results differ beyond tolerance")
        return 1

def main():
    """Main validation."""
    print("Loading workspace...")
    workspace = load_workspace()

    print("Fitting with pyhf...")
    pyhf_res = pyhf_fit(workspace)

    print("Fitting with NextStat...")
    nextstat_res = nextstat_fit(workspace)

    return compare_results(pyhf_res, nextstat_res)

if __name__ == "__main__":
    sys.exit(main())
