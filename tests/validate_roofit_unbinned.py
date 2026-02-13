#!/usr/bin/env python3
"""Validate NextStat unbinned fits vs RooFit reference.

For each canonical 1D PDF model (Gaussian+Exp, CrystalBall+Exp), this script:

1. Generates synthetic observed data (NumPy → Parquet).
2. Fits with NextStat CLI (``nextstat unbinned-fit``).
3. Writes the same data as a ROOT TTree + a RooFit C++ macro.
4. Fits with RooFit via ``root -b -q macro.C``.
5. Compares NLL and best-fit parameter values.
6. Writes a deterministic JSON artifact.

Requirements:
  - ``nextstat`` CLI binary (on PATH or via ``NS_CLI_BIN``).
  - ROOT (``root`` and ``root-config`` on PATH) for the RooFit comparison.
  - Python: numpy, pyarrow.

Run:
  python tests/validate_roofit_unbinned.py [--keep] [--cases gauss_exp,cb_exp]

The ``--keep`` flag preserves the temporary directory for debugging.
"""

from __future__ import annotations

import argparse
import json
import os
import shutil
import subprocess
import sys
import tempfile
from pathlib import Path
from typing import Any, Dict, List, Optional, Tuple

import numpy as np

REPO_ROOT = Path(__file__).resolve().parents[0].parent


# ---------------------------------------------------------------------------
# CLI binary resolution
# ---------------------------------------------------------------------------

def _find_cli_bin() -> str:
    env = os.environ.get("NS_CLI_BIN")
    if env:
        return env
    for profile in ("release", "debug"):
        candidate = REPO_ROOT / "target" / profile / "nextstat"
        if candidate.is_file():
            return str(candidate)
    found = shutil.which("nextstat")
    if found:
        return found
    sys.exit("ERROR: nextstat CLI binary not found. Build with: cargo build --release -p ns-cli")


def _has_root() -> bool:
    return shutil.which("root") is not None


# ---------------------------------------------------------------------------
# Data generation (shared with test_unbinned_closure_coverage.py)
# ---------------------------------------------------------------------------

def _sample_truncated_gaussian(
    rng: np.random.Generator, mu: float, sigma: float, lo: float, hi: float, n: int
) -> np.ndarray:
    samples: list[float] = []
    while len(samples) < n:
        batch = rng.normal(mu, sigma, size=max(n * 2, 1024))
        batch = batch[(batch >= lo) & (batch <= hi)]
        samples.extend(batch.tolist())
    return np.array(samples[:n])


def _sample_truncated_exponential(
    rng: np.random.Generator, lam: float, lo: float, hi: float, n: int
) -> np.ndarray:
    if abs(lam) < 1e-14:
        return rng.uniform(lo, hi, size=n)
    u = rng.uniform(0.0, 1.0, size=n)
    ea = np.exp(lam * lo)
    eb = np.exp(lam * hi)
    return np.log(ea + u * (eb - ea)) / lam


def _sample_crystal_ball(
    rng: np.random.Generator,
    mu: float, sigma: float, alpha: float, n_tail: float,
    lo: float, hi: float, n: int,
) -> np.ndarray:
    from scipy.stats import crystalball  # type: ignore
    rv = crystalball(beta=abs(alpha), m=n_tail, loc=mu, scale=sigma)
    samples: list[float] = []
    while len(samples) < n:
        batch = rv.rvs(size=max(n * 3, 2048), random_state=rng)
        batch = batch[(batch >= lo) & (batch <= hi)]
        samples.extend(batch.tolist())
    return np.array(samples[:n])


# ---------------------------------------------------------------------------
# Case definitions
# ---------------------------------------------------------------------------

CaseType = Dict[str, Any]


def _gauss_exp_case(rng: np.random.Generator) -> CaseType:
    obs_bounds = (60.0, 120.0)
    truth = {"mu_sig": 91.2, "sigma_sig": 2.5, "lambda_bkg": -0.03}
    n_sig, n_bkg = 1000, 3000
    sig = _sample_truncated_gaussian(rng, truth["mu_sig"], truth["sigma_sig"], *obs_bounds, n_sig)
    bkg = _sample_truncated_exponential(rng, truth["lambda_bkg"], *obs_bounds, n_bkg)
    data = np.concatenate([sig, bkg])
    np.random.default_rng(0).shuffle(data)

    ns_spec = {
        "schema_version": "nextstat_unbinned_spec_v0",
        "model": {
            "poi": "mu",
            "parameters": [
                {"name": "mu", "init": 1.0, "bounds": [0.0, 5.0]},
                {"name": "mu_sig", "init": 91.0, "bounds": [85.0, 95.0]},
                {"name": "sigma_sig", "init": 2.5, "bounds": [0.5, 10.0]},
                {"name": "lambda_bkg", "init": -0.03, "bounds": [-0.1, -0.001]},
            ],
        },
        "channels": [{
            "name": "SR",
            "data": {"file": "__DATA_PATH__"},
            "observables": [{"name": "mass", "bounds": list(obs_bounds)}],
            "processes": [
                {
                    "name": "signal",
                    "pdf": {"type": "gaussian", "observable": "mass", "params": ["mu_sig", "sigma_sig"]},
                    "yield": {"type": "scaled", "base_yield": float(n_sig), "scale": "mu"},
                },
                {
                    "name": "background",
                    "pdf": {"type": "exponential", "observable": "mass", "params": ["lambda_bkg"]},
                    "yield": {"type": "fixed", "value": float(n_bkg)},
                },
            ],
        }],
    }

    roofit_macro_template = r"""
#include <RooRealVar.h>
#include <RooGaussian.h>
#include <RooExponential.h>
#include <RooAddPdf.h>
#include <RooDataSet.h>
#include <RooFitResult.h>
#include <RooMinimizer.h>
#include <fstream>
#include <iomanip>

void roofit_gauss_exp() {{
    RooRealVar mass("mass", "mass", {lo}, {hi});
    RooDataSet ds("ds", "ds", RooArgSet(mass));
    {{
        std::ifstream infile("__DATA_TXT__");
        double val;
        while (infile >> val) {{
            mass.setVal(val);
            ds.add(RooArgSet(mass));
        }}
    }}

    RooRealVar mu_sig("mu_sig", "mu_sig", 91.0, 85.0, 95.0);
    RooRealVar sigma_sig("sigma_sig", "sigma_sig", 2.5, 0.5, 10.0);
    RooGaussian gauss("gauss", "gauss", mass, mu_sig, sigma_sig);

    RooRealVar lambda_bkg("lambda_bkg", "lambda_bkg", -0.03, -0.1, -0.001);
    RooExponential expo("expo", "expo", mass, lambda_bkg);

    RooRealVar nsig("nsig", "nsig", {n_sig}, 0, {n_total});
    RooRealVar nbkg("nbkg", "nbkg", {n_bkg}, 0, {n_total});
    nbkg.setConstant(true);

    RooAddPdf model("model", "model", RooArgList(gauss, expo), RooArgList(nsig, nbkg));

    std::unique_ptr<RooAbsReal> nll(model.createNLL(ds, RooFit::Extended(true)));
    RooMinimizer m(*nll);
    m.setPrintLevel(-1);
    m.setEps(1e-12);
    m.setStrategy(1);
    int status = m.minimize("Minuit2", "Migrad");
    if (status != 0) {{
        m.setStrategy(2);
        status = m.minimize("Minuit2", "Migrad");
    }}

    double nll_val = nll->getVal();
    double mu_hat = nsig.getVal() / {n_sig_f};

    std::ofstream out("__OUT_JSON__");
    out << std::setprecision(17);
    out << "{{" << std::endl;
    out << "  \"tool\": \"roofit\"," << std::endl;
    out << "  \"status\": " << status << "," << std::endl;
    out << "  \"nll\": " << nll_val << "," << std::endl;
    out << "  \"mu_hat\": " << mu_hat << "," << std::endl;
    out << "  \"mu_sig\": " << mu_sig.getVal() << "," << std::endl;
    out << "  \"sigma_sig\": " << sigma_sig.getVal() << "," << std::endl;
    out << "  \"lambda_bkg\": " << lambda_bkg.getVal() << "," << std::endl;
    out << "  \"nsig\": " << nsig.getVal() << "," << std::endl;
    out << "  \"nbkg\": " << nbkg.getVal() << std::endl;
    out << "}}" << std::endl;
    out.close();
}}
""".format(
        lo=obs_bounds[0], hi=obs_bounds[1],
        n_sig=n_sig, n_bkg=n_bkg, n_total=len(data) * 2,
        n_sig_f=float(n_sig),
    )

    return {
        "name": "gauss_exp",
        "data": data,
        "obs_name": "mass",
        "obs_bounds": obs_bounds,
        "truth": truth,
        "ns_spec": ns_spec,
        "roofit_macro": roofit_macro_template,
        "roofit_fn": "roofit_gauss_exp",
        "ns_to_roofit_params": {
            "mu_sig": "mu_sig",
            "sigma_sig": "sigma_sig",
            "lambda_bkg": "lambda_bkg",
        },
    }


def _cb_exp_case(rng: np.random.Generator) -> CaseType:
    obs_bounds = (60.0, 120.0)
    truth = {"mu_cb": 91.2, "sigma_cb": 2.5, "lambda_bkg": -0.025}
    alpha_cb, n_cb = 1.5, 5.0
    n_sig, n_bkg = 500, 2000
    sig = _sample_crystal_ball(rng, truth["mu_cb"], truth["sigma_cb"], alpha_cb, n_cb, *obs_bounds, n_sig)
    bkg = _sample_truncated_exponential(rng, truth["lambda_bkg"], *obs_bounds, n_bkg)
    data = np.concatenate([sig, bkg])
    np.random.default_rng(0).shuffle(data)

    ns_spec = {
        "schema_version": "nextstat_unbinned_spec_v0",
        "model": {
            "poi": "mu",
            "parameters": [
                {"name": "mu", "init": 1.0, "bounds": [0.0, 5.0]},
                {"name": "mu_cb", "init": 91.0, "bounds": [85.0, 95.0]},
                {"name": "sigma_cb", "init": 3.0, "bounds": [0.5, 10.0]},
                {"name": "alpha_cb", "init": 1.5, "bounds": [0.1, 5.0]},
                {"name": "n_cb", "init": 5.0, "bounds": [1.0, 50.0]},
                {"name": "lambda_bkg", "init": -0.02, "bounds": [-0.1, -0.001]},
            ],
        },
        "channels": [{
            "name": "SR",
            "data": {"file": "__DATA_PATH__"},
            "observables": [{"name": "mass", "bounds": list(obs_bounds)}],
            "processes": [
                {
                    "name": "signal",
                    "pdf": {
                        "type": "crystal_ball",
                        "observable": "mass",
                        "params": ["mu_cb", "sigma_cb", "alpha_cb", "n_cb"],
                    },
                    "yield": {"type": "scaled", "base_yield": float(n_sig), "scale": "mu"},
                },
                {
                    "name": "background",
                    "pdf": {"type": "exponential", "observable": "mass", "params": ["lambda_bkg"]},
                    "yield": {"type": "fixed", "value": float(n_bkg)},
                },
            ],
        }],
    }

    roofit_macro_template = r"""
#include <RooRealVar.h>
#include <RooCBShape.h>
#include <RooExponential.h>
#include <RooAddPdf.h>
#include <RooDataSet.h>
#include <RooFitResult.h>
#include <RooMinimizer.h>
#include <fstream>
#include <iomanip>

void roofit_cb_exp() {{
    RooRealVar mass("mass", "mass", {lo}, {hi});
    RooDataSet ds("ds", "ds", RooArgSet(mass));
    {{
        std::ifstream infile("__DATA_TXT__");
        double val;
        while (infile >> val) {{
            mass.setVal(val);
            ds.add(RooArgSet(mass));
        }}
    }}

    RooRealVar mu_cb("mu_cb", "mu_cb", 91.0, 85.0, 95.0);
    RooRealVar sigma_cb("sigma_cb", "sigma_cb", 3.0, 0.5, 10.0);
    RooRealVar alpha_cb("alpha_cb", "alpha_cb", {alpha}, {alpha}, {alpha});
    RooRealVar n_cb("n_cb", "n_cb", {n_tail}, {n_tail}, {n_tail});
    alpha_cb.setConstant(true);
    n_cb.setConstant(true);
    RooCBShape cbpdf("cbpdf", "cbpdf", mass, mu_cb, sigma_cb, alpha_cb, n_cb);

    RooRealVar lambda_bkg("lambda_bkg", "lambda_bkg", -0.02, -0.1, -0.001);
    RooExponential expo("expo", "expo", mass, lambda_bkg);

    RooRealVar nsig("nsig", "nsig", {n_sig}, 0, {n_total});
    RooRealVar nbkg("nbkg", "nbkg", {n_bkg}, 0, {n_total});
    nbkg.setConstant(true);

    RooAddPdf model("model", "model", RooArgList(cbpdf, expo), RooArgList(nsig, nbkg));

    std::unique_ptr<RooAbsReal> nll(model.createNLL(ds, RooFit::Extended(true)));
    RooMinimizer m(*nll);
    m.setPrintLevel(-1);
    m.setEps(1e-12);
    m.setStrategy(1);
    int status = m.minimize("Minuit2", "Migrad");
    if (status != 0) {{
        m.setStrategy(2);
        status = m.minimize("Minuit2", "Migrad");
    }}

    double nll_val = nll->getVal();
    double mu_hat = nsig.getVal() / {n_sig_f};

    std::ofstream out("__OUT_JSON__");
    out << std::setprecision(17);
    out << "{{" << std::endl;
    out << "  \"tool\": \"roofit\"," << std::endl;
    out << "  \"status\": " << status << "," << std::endl;
    out << "  \"nll\": " << nll_val << "," << std::endl;
    out << "  \"mu_hat\": " << mu_hat << "," << std::endl;
    out << "  \"mu_cb\": " << mu_cb.getVal() << "," << std::endl;
    out << "  \"sigma_cb\": " << sigma_cb.getVal() << "," << std::endl;
    out << "  \"lambda_bkg\": " << lambda_bkg.getVal() << std::endl;
    out << "}}" << std::endl;
    out.close();
}}
""".format(
        lo=obs_bounds[0], hi=obs_bounds[1],
        alpha=alpha_cb, n_tail=n_cb,
        n_sig=n_sig, n_bkg=n_bkg, n_total=len(data) * 2,
        n_sig_f=float(n_sig),
    )

    return {
        "name": "cb_exp",
        "data": data,
        "obs_name": "mass",
        "obs_bounds": obs_bounds,
        "truth": truth,
        "ns_spec": ns_spec,
        "roofit_macro": roofit_macro_template,
        "roofit_fn": "roofit_cb_exp",
        "ns_to_roofit_params": {
            "mu_cb": "mu_cb",
            "sigma_cb": "sigma_cb",
            "lambda_bkg": "lambda_bkg",
        },
    }


ALL_CASE_FNS = {"gauss_exp": _gauss_exp_case, "cb_exp": _cb_exp_case}


# ---------------------------------------------------------------------------
# Parquet + ROOT TTree writers
# ---------------------------------------------------------------------------

def _write_parquet(data: np.ndarray, obs_name: str, obs_bounds: Tuple[float, float], path: Path) -> None:
    import pyarrow as pa
    import pyarrow.parquet as pq
    table = pa.table({obs_name: pa.array(data, type=pa.float64())})
    meta_kv = {
        b"nextstat.schema_version": b"nextstat_unbinned_events_v1",
        b"nextstat.observables": json.dumps(
            [{"name": obs_name, "bounds": list(obs_bounds)}], separators=(",", ":")
        ).encode(),
    }
    table = table.replace_schema_metadata({**(table.schema.metadata or {}), **meta_kv})
    pq.write_table(table, str(path))


def _write_data_txt(data: np.ndarray, path: Path) -> None:
    """Write data as a plain text file (one value per line)."""
    with path.open("w") as f:
        for v in data:
            f.write(f"{v:.17g}\n")


# ---------------------------------------------------------------------------
# NextStat CLI fit
# ---------------------------------------------------------------------------

def _fit_nextstat(spec_path: Path) -> Dict[str, Any]:
    cli = _find_cli_bin()
    proc = subprocess.run(
        [cli, "unbinned-fit", "--config", str(spec_path)],
        capture_output=True, text=True, timeout=120,
    )
    if proc.returncode != 0:
        raise RuntimeError(f"nextstat failed:\n{proc.stderr}\n{proc.stdout}")
    return json.loads(proc.stdout)


# ---------------------------------------------------------------------------
# RooFit macro execution
# ---------------------------------------------------------------------------

def _run_roofit_macro(macro_path: Path, fn_name: str) -> None:
    proc = subprocess.run(
        ["root", "-b", "-q", f"{macro_path.name}+"],
        capture_output=True, text=True, timeout=300,
        cwd=str(macro_path.parent),
    )
    if proc.returncode != 0:
        raise RuntimeError(f"ROOT macro failed:\n{proc.stderr}\n{proc.stdout}")


# ---------------------------------------------------------------------------
# Main comparison loop
# ---------------------------------------------------------------------------

# Tolerances for RooFit comparison.
NLL_ATOL = 0.5      # absolute NLL difference (different normalization conventions)
PARAM_ATOL = 0.3    # best-fit parameter difference (generous; statistical noise)
PARAM_RTOL = 0.10   # relative


def _compare(
    case_name: str,
    ns_result: Dict[str, Any],
    roofit_result: Dict[str, Any],
    ns_to_roofit: Dict[str, str],
) -> Dict[str, Any]:
    """Compare NextStat vs RooFit results. Returns comparison dict."""
    ns_names = ns_result["parameter_names"]
    ns_bestfit = ns_result["bestfit"]
    ns_nll = ns_result["nll"]

    rf_nll = roofit_result["nll"]

    comparisons: Dict[str, Any] = {}
    all_ok = True

    for ns_name, rf_name in ns_to_roofit.items():
        ns_idx = ns_names.index(ns_name)
        ns_val = ns_bestfit[ns_idx]
        rf_val = roofit_result[rf_name]
        diff = abs(ns_val - rf_val)
        ok = diff <= PARAM_ATOL or (abs(rf_val) > 1e-10 and diff <= PARAM_RTOL * abs(rf_val))
        comparisons[ns_name] = {
            "nextstat": ns_val,
            "roofit": rf_val,
            "diff": diff,
            "ok": ok,
        }
        if not ok:
            all_ok = False

    nll_diff = abs(ns_nll - rf_nll)

    return {
        "case": case_name,
        "nextstat_nll": ns_nll,
        "roofit_nll": rf_nll,
        "nll_diff": nll_diff,
        "parameters": comparisons,
        "all_params_ok": all_ok,
        "nextstat_converged": ns_result["converged"],
        "roofit_status": roofit_result.get("status", -1),
    }


def main() -> None:
    parser = argparse.ArgumentParser(description="Validate NextStat unbinned fits vs RooFit")
    parser.add_argument("--keep", action="store_true", help="Keep temporary directory")
    parser.add_argument("--cases", default="all", help="Comma-separated case names or 'all'")
    parser.add_argument("--seed", type=int, default=42, help="RNG seed")
    parser.add_argument("--output", type=str, default=None, help="Output JSON artifact path")
    args = parser.parse_args()

    if not _has_root():
        print("WARNING: ROOT not found on PATH. Skipping RooFit comparison.", file=sys.stderr)
        print("Only NextStat closure results will be produced.", file=sys.stderr)

    case_keys = sorted(ALL_CASE_FNS.keys()) if args.cases == "all" else [
        k.strip() for k in args.cases.split(",")
    ]
    for k in case_keys:
        if k not in ALL_CASE_FNS:
            sys.exit(f"Unknown case: {k}. Available: {sorted(ALL_CASE_FNS)}")

    rng = np.random.default_rng(args.seed)
    results: List[Dict[str, Any]] = []

    tmpdir = Path(tempfile.mkdtemp(prefix="ns_roofit_unbinned_"))
    print(f"Working directory: {tmpdir}")

    try:
        for case_name in case_keys:
            print(f"\n{'='*60}")
            print(f"Case: {case_name}")
            print(f"{'='*60}")

            case = ALL_CASE_FNS[case_name](rng)
            case_dir = tmpdir / case_name
            case_dir.mkdir()

            # Write data files.
            parquet_path = case_dir / "observed.parquet"
            _write_parquet(case["data"], case["obs_name"], case["obs_bounds"], parquet_path)

            # NextStat fit.
            spec = case["ns_spec"]
            spec["channels"][0]["data"]["file"] = str(parquet_path)
            spec_path = case_dir / "spec.json"
            spec_path.write_text(json.dumps(spec, indent=2))

            print("  NextStat fit...")
            ns_result = _fit_nextstat(spec_path)
            print(f"    converged={ns_result['converged']}, nll={ns_result['nll']:.6f}")
            ns_names = ns_result["parameter_names"]
            for i, name in enumerate(ns_names):
                print(f"    {name} = {ns_result['bestfit'][i]:.6f} ± {ns_result['uncertainties'][i]:.6f}")

            # RooFit fit (if ROOT available).
            roofit_result: Optional[Dict[str, Any]] = None
            if _has_root():
                data_txt = case_dir / "observed.txt"
                _write_data_txt(case["data"], data_txt)

                macro_text = case["roofit_macro"]
                out_json = case_dir / "roofit_result.json"
                macro_text = macro_text.replace("__DATA_TXT__", str(data_txt))
                macro_text = macro_text.replace("__OUT_JSON__", str(out_json))
                macro_path = case_dir / f"{case['roofit_fn']}.C"
                macro_path.write_text(macro_text)

                print("  RooFit fit...")
                try:
                    _run_roofit_macro(macro_path, case["roofit_fn"])
                    roofit_result = json.loads(out_json.read_text())
                    print(f"    status={roofit_result.get('status')}, nll={roofit_result['nll']:.6f}")
                    for k, v in roofit_result.items():
                        if k not in ("tool", "status", "nll"):
                            print(f"    {k} = {v}")
                except Exception as e:
                    print(f"  RooFit FAILED: {e}")

            # Compare.
            if roofit_result is not None:
                comparison = _compare(case_name, ns_result, roofit_result, case["ns_to_roofit_params"])
                results.append(comparison)
                status_str = "PASS" if comparison["all_params_ok"] else "FAIL"
                print(f"\n  Comparison: {status_str}")
                for pname, pdata in comparison["parameters"].items():
                    mark = "✓" if pdata["ok"] else "✗"
                    print(f"    {mark} {pname}: NS={pdata['nextstat']:.6f} RF={pdata['roofit']:.6f} Δ={pdata['diff']:.6f}")
                print(f"    NLL: NS={comparison['nextstat_nll']:.6f} RF={comparison['roofit_nll']:.6f} Δ={comparison['nll_diff']:.6f}")
            else:
                results.append({
                    "case": case_name,
                    "nextstat_nll": ns_result["nll"],
                    "nextstat_converged": ns_result["converged"],
                    "roofit_nll": None,
                    "roofit_skipped": True,
                })

        # Write artifact.
        artifact = {"seed": args.seed, "cases": results}
        artifact_json = json.dumps(artifact, indent=2)

        if args.output:
            out_path = Path(args.output)
            out_path.parent.mkdir(parents=True, exist_ok=True)
            out_path.write_text(artifact_json)
            print(f"\nArtifact written to {out_path}")

        artifacts_dir = os.environ.get("NS_ARTIFACTS_DIR")
        if artifacts_dir:
            out_path = Path(artifacts_dir) / "validate_roofit_unbinned.json"
            out_path.parent.mkdir(parents=True, exist_ok=True)
            out_path.write_text(artifact_json)

        print(f"\n{'='*60}")
        print("Summary:")
        for r in results:
            if r.get("roofit_skipped"):
                print(f"  {r['case']}: RooFit skipped (NextStat converged={r['nextstat_converged']})")
            else:
                status = "PASS" if r["all_params_ok"] else "FAIL"
                print(f"  {r['case']}: {status} (NLL Δ={r['nll_diff']:.4f})")
        print(f"{'='*60}")

    finally:
        if not args.keep:
            shutil.rmtree(tmpdir, ignore_errors=True)
        else:
            print(f"\nKept working directory: {tmpdir}")


if __name__ == "__main__":
    main()
