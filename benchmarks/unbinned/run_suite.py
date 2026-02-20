#!/usr/bin/env python3
"""
Cross-framework unbinned benchmark suite.

This is intentionally a *harness*:
- it generates deterministic toy datasets
- runs NextStat via the CLI (required)
- optionally runs RooFit/zfit/MoreFit if available
- writes a single JSON artifact with timings + results

Notes:
- NLL normalizations differ across tools; treat NLL comparisons as indicative.
- This suite is separate from Criterion micro-benchmarks in ns-unbinned.
"""

from __future__ import annotations

import argparse
import importlib.util
import json
import multiprocessing
import os
import shutil
import subprocess
import sys
import tempfile
import time
from dataclasses import dataclass
from pathlib import Path
from typing import Any, Dict, List, Optional, Tuple

import numpy as np

sys.path.insert(0, os.path.join(os.path.dirname(__file__), "../../scripts/benchmarks"))
from _parse_utils import parse_json_stdout  # noqa: E402


REPO_ROOT = Path(__file__).resolve().parents[2]


def _die(msg: str) -> "NoReturn":  # type: ignore[name-defined]
    raise SystemExit(msg)


def _find_nextstat_cli() -> str:
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
    _die("ERROR: nextstat CLI not found. Build with: cargo build --release -p ns-cli")


def _has_root() -> bool:
    return shutil.which("root") is not None


def _has_import(mod: str) -> bool:
    # Avoid importing heavy dependencies just to check availability.
    return importlib.util.find_spec(mod) is not None


def _now_ms() -> float:
    return time.perf_counter() * 1000.0


def _as_float(x: Any) -> float:
    # zfit/TF values, numpy scalars, python numbers
    try:
        if hasattr(x, "numpy"):
            return float(x.numpy())
    except Exception:
        pass
    try:
        return float(x)
    except Exception:
        return float("nan")


def _render_tokens(s: str, mapping: Dict[str, Any]) -> str:
    # Simple, explicit token replacement (avoids brace-escaping issues in C++ macros).
    for k, v in mapping.items():
        s = s.replace(k, str(v))
    return s


def _sample_truncated_gaussian(
    rng: np.random.Generator, mu: float, sigma: float, lo: float, hi: float, n: int
) -> np.ndarray:
    samples: list[float] = []
    while len(samples) < n:
        batch = rng.normal(mu, sigma, size=max(n * 2, 2048))
        batch = batch[(batch >= lo) & (batch <= hi)]
        samples.extend(batch.tolist())
    return np.array(samples[:n], dtype=np.float64)


def _sample_truncated_exponential(
    rng: np.random.Generator, lam: float, lo: float, hi: float, n: int
) -> np.ndarray:
    if abs(lam) < 1e-14:
        return rng.uniform(lo, hi, size=n).astype(np.float64)
    u = rng.uniform(0.0, 1.0, size=n)
    ea = np.exp(lam * lo)
    eb = np.exp(lam * hi)
    return (np.log(ea + u * (eb - ea)) / lam).astype(np.float64)


def _sample_crystal_ball(
    rng: np.random.Generator,
    mu: float,
    sigma: float,
    alpha: float,
    n_tail: float,
    lo: float,
    hi: float,
    n: int,
) -> np.ndarray:
    # Optional dependency; used only for this case.
    try:
        from scipy.stats import crystalball  # type: ignore
    except Exception as e:
        raise RuntimeError("scipy is required for the cb_exp case (scipy.stats.crystalball)") from e

    rv = crystalball(beta=abs(alpha), m=n_tail, loc=mu, scale=sigma)
    samples: list[float] = []
    while len(samples) < n:
        batch = rv.rvs(size=max(n * 3, 4096), random_state=rng)
        batch = batch[(batch >= lo) & (batch <= hi)]
        samples.extend(batch.tolist())
    return np.array(samples[:n], dtype=np.float64)


def _write_parquet(
    *,
    columns: Dict[str, np.ndarray],
    observables: List[Dict[str, Any]],
    path: Path,
) -> None:
    try:
        import pyarrow as pa  # type: ignore
        import pyarrow.parquet as pq  # type: ignore
    except Exception as e:
        raise RuntimeError("pyarrow is required to write Parquet inputs") from e

    table = pa.table({k: pa.array(v, type=pa.float64()) for k, v in columns.items()})
    meta_kv = {
        b"nextstat.schema_version": b"nextstat_unbinned_events_v1",
        b"nextstat.observables": json.dumps(observables, separators=(",", ":")).encode(),
    }
    table = table.replace_schema_metadata({**(table.schema.metadata or {}), **meta_kv})
    pq.write_table(table, str(path))


def _fit_nextstat(spec_path: Path) -> Dict[str, Any]:
    cli = _find_nextstat_cli()
    t0 = _now_ms()
    proc = subprocess.run(
        [cli, "unbinned-fit", "--config", str(spec_path)],
        capture_output=True,
        text=True,
        timeout=180,
    )
    wall_ms = _now_ms() - t0
    if proc.returncode != 0:
        raise RuntimeError(f"nextstat failed:\n{proc.stderr}\n{proc.stdout}")
    out = parse_json_stdout(proc.stdout)
    out["_wall_ms"] = wall_ms
    return out


def _run_root_macro(macro_path: Path, fn_name: str) -> Dict[str, Any]:
    if not _has_root():
        return {"skipped": True, "reason": "root not found on PATH"}

    t0 = _now_ms()
    proc = subprocess.run(
        ["root", "-b", "-q", f"{macro_path.name}+"],
        cwd=str(macro_path.parent),
        capture_output=True,
        text=True,
        timeout=300,
    )
    wall_ms = _now_ms() - t0
    if proc.returncode != 0:
        return {
            "failed": True,
            "reason": f"root returned {proc.returncode}",
            "stderr": proc.stderr[-4000:],
            "stdout": proc.stdout[-4000:],
            "_wall_ms": wall_ms,
        }

    out_json = macro_path.parent / "roofit_result.json"
    if not out_json.is_file():
        return {
            "failed": True,
            "reason": f"missing output json: {out_json}",
            "stderr": proc.stderr[-4000:],
            "stdout": proc.stdout[-4000:],
            "_wall_ms": wall_ms,
        }
    out = json.loads(out_json.read_text())
    out["_wall_ms"] = wall_ms
    # Separate fit time from process overhead (JIT compilation, startup, I/O)
    fit_real_ms = out.get("fit_real_s", 0) * 1000.0
    out["_overhead_ms"] = round(wall_ms - fit_real_ms, 1)
    out["_fn"] = fn_name
    return out


def _fit_zfit(
    *,
    case_name: str,
    data: np.ndarray,
    bounds: Tuple[float, float] | Tuple[Tuple[float, float], Tuple[float, float]],
    model_kind: str,
    init: Dict[str, float],
    limits: Dict[str, Tuple[float, float]],
    constants: Dict[str, float],
) -> Dict[str, Any]:
    if not _has_import("zfit"):
        return {"skipped": True, "reason": "zfit not importable"}

    import zfit  # type: ignore

    # zfit is TF-backed; keep the code defensive against small API differences.
    t0 = _now_ms()

    # Observables
    if model_kind in ("gauss_exp_ext", "cb_exp_ext"):
        lo, hi = bounds  # type: ignore[misc]
        obs = zfit.Space("mass", (lo, hi))
    elif model_kind == "product2d":
        (xlo, xhi), (ylo, yhi) = bounds  # type: ignore[misc]
        # zfit expects multi-observable limits as ((lower_x, lower_y), (upper_x, upper_y)).
        obs = zfit.Space(["x", "y"], limits=((xlo, ylo), (xhi, yhi)))
    else:
        return {"skipped": True, "reason": f"unknown model_kind={model_kind}"}

    def P(name: str) -> Any:
        v = float(init[name])
        lo_, hi_ = limits[name]
        return zfit.Parameter(name, v, lo_, hi_)

    def C(name: str) -> Any:
        v = float(constants[name])
        # zfit ≥0.28 rejects lo==hi in Parameter(); prefer ConstantParameter.
        # Fallback order: ConstantParameter → Parameter(floating=False, wide bounds) → raw float.
        try:
            return zfit.param.ConstantParameter(name, v)
        except Exception:
            pass
        try:
            # Give wide bounds so zfit doesn't reject lo==hi or value outside range.
            margin = max(abs(v) * 10.0, 100.0)
            p = zfit.Parameter(name, v, v - margin, v + margin, floating=False)
            return p
        except Exception:
            pass
        # Last resort: zfit PDFs accept raw floats for constant arguments.
        return v

    # Build PDF(s)
    params: Dict[str, Any] = {}

    if model_kind == "gauss_exp_ext":
        params["mu_sig"] = P("mu_sig")
        params["sigma_sig"] = P("sigma_sig")
        params["lambda_bkg"] = P("lambda_bkg")
        params["nsig"] = P("nsig")
        params["nbkg"] = C("nbkg")

        gauss = zfit.pdf.Gauss(obs=obs, mu=params["mu_sig"], sigma=params["sigma_sig"]).create_extended(
            params["nsig"]
        )
        expo = zfit.pdf.Exponential(obs=obs, lam=params["lambda_bkg"]).create_extended(params["nbkg"])
        model = zfit.pdf.SumPDF([gauss, expo])
        loss = zfit.loss.ExtendedUnbinnedNLL(model=model, data=_zfit_data(obs, data))

    elif model_kind == "cb_exp_ext":
        params["mu_cb"] = P("mu_cb")
        params["sigma_cb"] = P("sigma_cb")
        params["lambda_bkg"] = P("lambda_bkg")
        params["nsig"] = P("nsig")
        params["nbkg"] = C("nbkg")
        params["alpha_cb"] = C("alpha_cb")
        params["n_cb"] = C("n_cb")

        cb = zfit.pdf.CrystalBall(
            mu=params["mu_cb"],
            sigma=params["sigma_cb"],
            alpha=params["alpha_cb"],
            n=params["n_cb"],
            obs=obs,
        ).create_extended(params["nsig"])
        expo = zfit.pdf.Exponential(obs=obs, lam=params["lambda_bkg"]).create_extended(params["nbkg"])
        model = zfit.pdf.SumPDF([cb, expo])
        loss = zfit.loss.ExtendedUnbinnedNLL(model=model, data=_zfit_data(obs, data))

    elif model_kind == "product2d":
        params["x_mu"] = P("x_mu")
        params["x_sigma"] = P("x_sigma")
        params["y_lambda"] = P("y_lambda")

        # Build 1D PDFs on dedicated spaces to allow factorized integration.
        obs_x = zfit.Space("x", (bounds[0][0], bounds[0][1]))  # type: ignore[index]
        obs_y = zfit.Space("y", (bounds[1][0], bounds[1][1]))  # type: ignore[index]
        gx = zfit.pdf.Gauss(obs=obs_x, mu=params["x_mu"], sigma=params["x_sigma"])
        ey = zfit.pdf.Exponential(obs=obs_y, lam=params["y_lambda"])
        model = zfit.pdf.ProductPDF([gx, ey], obs=obs)
        loss = zfit.loss.UnbinnedNLL(model=model, data=_zfit_data(obs, data))

    else:
        return {"skipped": True, "reason": f"unknown model_kind={model_kind}"}

    # Minimize
    minimizer = zfit.minimize.Minuit()
    try:
        result = minimizer.minimize(loss)
        try:
            result = result.update_params()
        except Exception:
            pass
    except Exception as e:
        return {"failed": True, "reason": f"zfit minimize failed: {e}", "_wall_ms": _now_ms() - t0}

    # Errors (prefer hesse; weights semantics are tricky)
    errors: Dict[str, float] = {}
    try:
        h = result.hesse()
        for p, err in h.items():
            errors[getattr(p, "name", str(p))] = _as_float(err)
    except Exception:
        try:
            h = result.hesse(method="hesse_np")
            for p, err in h.items():
                errors[getattr(p, "name", str(p))] = _as_float(err)
        except Exception:
            pass

    # Extract values
    values: Dict[str, float] = {}
    for name, p in params.items():
        try:
            values[name] = _as_float(p.value())
        except Exception:
            try:
                values[name] = _as_float(p)
            except Exception:
                values[name] = float("nan")

    nll_val = float("nan")
    try:
        nll_val = _as_float(loss.value())
    except Exception:
        try:
            nll_val = _as_float(result.fmin)
        except Exception:
            pass

    return {
        "tool": "zfit",
        "case": case_name,
        "converged": bool(getattr(result, "converged", True)),
        "valid": bool(getattr(result, "valid", True)),
        "nll": nll_val,
        "values": values,
        "errors": errors,
        "_wall_ms": _now_ms() - t0,
    }


def _zfit_data(obs: Any, data: np.ndarray) -> Any:
    import zfit  # type: ignore

    # Try a couple of common constructors.
    try:
        return zfit.Data.from_numpy(obs=obs, array=data)
    except Exception:
        pass
    try:
        return zfit.Data.from_tensor(obs=obs, tensor=data)
    except Exception:
        pass
    return zfit.Data(obs=obs, data=data)


_ZFIT_TIMEOUT_S = 120


def _zfit_worker(q: multiprocessing.Queue, kwargs: dict) -> None:
    """Target for the spawned zfit subprocess."""
    try:
        result = _fit_zfit(**kwargs)
        q.put(result)
    except Exception as e:
        q.put({"failed": True, "reason": f"zfit subprocess error: {e}"})


def _fit_zfit_with_timeout(timeout_s: int = _ZFIT_TIMEOUT_S, **kwargs) -> dict:
    """Run _fit_zfit in a separate process with a hard timeout.

    zfit is TF-backed and may hang on certain models/data. This wrapper
    guarantees the benchmark harness will not block indefinitely.
    """
    if not _has_import("zfit"):
        return {"skipped": True, "reason": "zfit not importable"}

    ctx = multiprocessing.get_context("spawn")
    q = ctx.Queue()
    p = ctx.Process(target=_zfit_worker, args=(q, kwargs))
    p.start()
    p.join(timeout=timeout_s)
    if p.is_alive():
        p.terminate()
        p.join(timeout=5)
        if p.is_alive():
            p.kill()
        return {"failed": True, "reason": f"zfit timeout after {timeout_s}s"}
    if q.empty():
        return {"failed": True, "reason": "zfit subprocess exited without result"}
    return q.get_nowait()


def _fit_morefit(
    *,
    case_name: str,
    data: np.ndarray,
    bounds: Tuple[float, float],
    model_kind: str,
    tmpdir: Path,
) -> Dict[str, Any]:
    """Run MoreFit via its C++ binary (subprocess, like RooFit).

    Requires ``morefit_gauss_exp`` binary on PATH or at MOREFIT_BIN env var.
    Only supports gauss_exp (fraction-based Gauss + Exp, 4 free params).
    """
    if model_kind not in ("gauss_exp_ext",):
        return {"skipped": True, "reason": f"morefit runner only supports gauss_exp, got {model_kind}"}

    binary = os.environ.get("MOREFIT_BIN", shutil.which("morefit_gauss_exp") or "")
    if not binary or not Path(binary).is_file():
        return {"skipped": True, "reason": "morefit_gauss_exp binary not found (set MOREFIT_BIN)"}

    data_path = tmpdir / f"morefit_{case_name}.txt"
    np.savetxt(str(data_path), data, fmt="%.17g")

    lo, hi = bounds
    t0 = _now_ms()
    try:
        proc = subprocess.run(
            [binary, str(data_path), str(lo), str(hi), "3"],
            capture_output=True,
            text=True,
            timeout=120,
        )
    except subprocess.TimeoutExpired:
        return {"failed": True, "reason": "morefit timed out (120s)", "case": case_name}

    wall_ms = _now_ms() - t0

    if proc.returncode != 0:
        return {"failed": True, "reason": f"morefit exited {proc.returncode}: {proc.stderr[:300]}", "case": case_name}

    stdout = proc.stdout.strip()
    json_start = stdout.rfind("\n{")
    if json_start >= 0:
        json_start += 1  # skip the newline
    else:
        json_start = stdout.rfind("{")
    if json_start < 0:
        return {"failed": True, "reason": "no JSON in morefit output", "case": case_name}

    try:
        result = json.loads(stdout[json_start:])
    except json.JSONDecodeError as e:
        return {"failed": True, "reason": f"morefit JSON parse error: {e}", "case": case_name}

    result["tool"] = "morefit"
    result["case"] = case_name
    result["_wall_ms"] = wall_ms
    return result


@dataclass(frozen=True)
class Case:
    name: str
    kind: str
    n_events: int


def _case_gauss_exp(rng: np.random.Generator, n_events: int) -> Dict[str, Any]:
    obs_bounds = (60.0, 120.0)
    n_sig = max(1, n_events // 4)
    n_bkg = max(1, n_events - n_sig)
    truth = {"mu_sig": 91.2, "sigma_sig": 2.5, "lambda_bkg": -0.03}

    sig = _sample_truncated_gaussian(rng, truth["mu_sig"], truth["sigma_sig"], *obs_bounds, n_sig)
    bkg = _sample_truncated_exponential(rng, truth["lambda_bkg"], *obs_bounds, n_bkg)
    data = np.concatenate([sig, bkg])
    rng.shuffle(data)

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
        "channels": [
            {
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
            }
        ],
    }

    roofit_macro_tmpl = r"""
#include <RooRealVar.h>
#include <RooGaussian.h>
#include <RooExponential.h>
#include <RooAddPdf.h>
#include <RooDataSet.h>
#include <RooMinimizer.h>
#include <TStopwatch.h>
#include <memory>
#include <fstream>
#include <iomanip>

void roofit_gauss_exp() {
  RooRealVar mass("mass", "mass", __LO__, __HI__);
  RooDataSet ds("ds", "ds", RooArgSet(mass));
  {
    std::ifstream infile("__DATA_TXT__");
    double val;
    while (infile >> val) {
      mass.setVal(val);
      ds.add(RooArgSet(mass));
    }
  }

  RooRealVar mu_sig("mu_sig", "mu_sig", 91.0, 85.0, 95.0);
  RooRealVar sigma_sig("sigma_sig", "sigma_sig", 2.5, 0.5, 10.0);
  RooGaussian gauss("gauss", "gauss", mass, mu_sig, sigma_sig);

  RooRealVar lambda_bkg("lambda_bkg", "lambda_bkg", -0.03, -0.1, -0.001);
  RooExponential expo("expo", "expo", mass, lambda_bkg);

  RooRealVar nsig("nsig", "nsig", __N_SIG__, 0, __N_TOTAL__);
  RooRealVar nbkg("nbkg", "nbkg", __N_BKG__, 0, __N_TOTAL__);
  nbkg.setConstant(true);

  RooAddPdf model("model", "model", RooArgList(gauss, expo), RooArgList(nsig, nbkg));

  TStopwatch sw;
  sw.Start();
  std::unique_ptr<RooAbsReal> nll(model.createNLL(ds, RooFit::Extended(true)));
  RooMinimizer m(*nll);
  m.setPrintLevel(-1);
  m.setEps(1e-12);
  m.setStrategy(1);
  int status = m.minimize("Minuit2", "Migrad");
  if (status != 0) {
    m.setStrategy(2);
    status = m.minimize("Minuit2", "Migrad");
  }
  m.hesse();
  sw.Stop();

  double nll_val = nll->getVal();
  double mu_hat = nsig.getVal() / __N_SIG_F__;
  double fit_real_s = sw.RealTime();
  double fit_cpu_s = sw.CpuTime();

  std::ofstream out("roofit_result.json");
  out << std::setprecision(17);
  out << "{\n";
  out << "  \"tool\": \"roofit\",\n";
  out << "  \"status\": " << status << ",\n";
  out << "  \"nll\": " << nll_val << ",\n";
  out << "  \"mu_hat\": " << mu_hat << ",\n";
  out << "  \"fit_real_s\": " << fit_real_s << ",\n";
  out << "  \"fit_cpu_s\": " << fit_cpu_s << ",\n";
  out << "  \"mu_sig\": " << mu_sig.getVal() << ",\n";
  out << "  \"mu_sig_err\": " << mu_sig.getError() << ",\n";
  out << "  \"sigma_sig\": " << sigma_sig.getVal() << ",\n";
  out << "  \"sigma_sig_err\": " << sigma_sig.getError() << ",\n";
  out << "  \"lambda_bkg\": " << lambda_bkg.getVal() << ",\n";
  out << "  \"lambda_bkg_err\": " << lambda_bkg.getError() << "\n";
  out << "}\n";
  out.close();
}
"""
    roofit_macro = _render_tokens(
        roofit_macro_tmpl,
        {
            "__LO__": obs_bounds[0],
            "__HI__": obs_bounds[1],
            "__N_SIG__": n_sig,
            "__N_BKG__": n_bkg,
            "__N_TOTAL__": int(n_events * 2),
            "__N_SIG_F__": float(n_sig),
        },
    )

    return {
        "kind": "gauss_exp_ext",
        "observables": [{"name": "mass", "bounds": list(obs_bounds)}],
        "columns": {"mass": data},
        "ns_spec": ns_spec,
        "roofit": {"fn": "roofit_gauss_exp", "macro": roofit_macro},
        "zfit": {
            "model_kind": "gauss_exp_ext",
            "bounds": obs_bounds,
            "data": data,
            "init": {"mu_sig": 91.0, "sigma_sig": 2.5, "lambda_bkg": -0.03, "nsig": float(n_sig), "nbkg": float(n_bkg)},
            "limits": {
                "mu_sig": (85.0, 95.0),
                "sigma_sig": (0.5, 10.0),
                "lambda_bkg": (-0.1, -0.001),
                "nsig": (0.0, float(n_events * 2)),
                "nbkg": (float(n_bkg), float(n_bkg)),
            },
            "constants": {"nbkg": float(n_bkg)},
        },
    }


def _case_cb_exp(rng: np.random.Generator, n_events: int) -> Dict[str, Any]:
    obs_bounds = (60.0, 120.0)
    alpha_cb, n_cb = 1.5, 5.0
    n_sig = max(1, n_events // 5)
    n_bkg = max(1, n_events - n_sig)
    truth = {"mu_cb": 91.2, "sigma_cb": 2.5, "lambda_bkg": -0.025}

    sig = _sample_crystal_ball(rng, truth["mu_cb"], truth["sigma_cb"], alpha_cb, n_cb, *obs_bounds, n_sig)
    bkg = _sample_truncated_exponential(rng, truth["lambda_bkg"], *obs_bounds, n_bkg)
    data = np.concatenate([sig, bkg])
    rng.shuffle(data)

    ns_spec = {
        "schema_version": "nextstat_unbinned_spec_v0",
        "model": {
            "poi": "mu",
            "parameters": [
                {"name": "mu", "init": 1.0, "bounds": [0.0, 5.0]},
                {"name": "mu_cb", "init": 91.0, "bounds": [85.0, 95.0]},
                {"name": "sigma_cb", "init": 3.0, "bounds": [0.5, 10.0]},
                {"name": "alpha_cb", "init": alpha_cb, "bounds": [alpha_cb, alpha_cb]},
                {"name": "n_cb", "init": n_cb, "bounds": [n_cb, n_cb]},
                {"name": "lambda_bkg", "init": -0.02, "bounds": [-0.1, -0.001]},
            ],
        },
        "channels": [
            {
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
            }
        ],
    }

    roofit_macro_tmpl = r"""
#include <RooRealVar.h>
#include <RooCBShape.h>
#include <RooExponential.h>
#include <RooAddPdf.h>
#include <RooDataSet.h>
#include <RooMinimizer.h>
#include <TStopwatch.h>
#include <memory>
#include <fstream>
#include <iomanip>

void roofit_cb_exp() {
  RooRealVar mass("mass", "mass", __LO__, __HI__);
  RooDataSet ds("ds", "ds", RooArgSet(mass));
  {
    std::ifstream infile("__DATA_TXT__");
    double val;
    while (infile >> val) {
      mass.setVal(val);
      ds.add(RooArgSet(mass));
    }
  }

  RooRealVar mu_cb("mu_cb", "mu_cb", 91.0, 85.0, 95.0);
  RooRealVar sigma_cb("sigma_cb", "sigma_cb", 3.0, 0.5, 10.0);
  RooRealVar alpha_cb("alpha_cb", "alpha_cb", __ALPHA__, __ALPHA__, __ALPHA__);
  RooRealVar n_cb("n_cb", "n_cb", __N_TAIL__, __N_TAIL__, __N_TAIL__);
  alpha_cb.setConstant(true);
  n_cb.setConstant(true);
  RooCBShape cbpdf("cbpdf", "cbpdf", mass, mu_cb, sigma_cb, alpha_cb, n_cb);

  RooRealVar lambda_bkg("lambda_bkg", "lambda_bkg", -0.02, -0.1, -0.001);
  RooExponential expo("expo", "expo", mass, lambda_bkg);

  RooRealVar nsig("nsig", "nsig", __N_SIG__, 0, __N_TOTAL__);
  RooRealVar nbkg("nbkg", "nbkg", __N_BKG__, 0, __N_TOTAL__);
  nbkg.setConstant(true);

  RooAddPdf model("model", "model", RooArgList(cbpdf, expo), RooArgList(nsig, nbkg));

  TStopwatch sw;
  sw.Start();
  std::unique_ptr<RooAbsReal> nll(model.createNLL(ds, RooFit::Extended(true)));
  RooMinimizer m(*nll);
  m.setPrintLevel(-1);
  m.setEps(1e-12);
  m.setStrategy(1);
  int status = m.minimize("Minuit2", "Migrad");
  if (status != 0) {
    m.setStrategy(2);
    status = m.minimize("Minuit2", "Migrad");
  }
  m.hesse();
  sw.Stop();

  double nll_val = nll->getVal();
  double mu_hat = nsig.getVal() / __N_SIG_F__;
  double fit_real_s = sw.RealTime();
  double fit_cpu_s = sw.CpuTime();

  std::ofstream out("roofit_result.json");
  out << std::setprecision(17);
  out << "{\n";
  out << "  \"tool\": \"roofit\",\n";
  out << "  \"status\": " << status << ",\n";
  out << "  \"nll\": " << nll_val << ",\n";
  out << "  \"mu_hat\": " << mu_hat << ",\n";
  out << "  \"fit_real_s\": " << fit_real_s << ",\n";
  out << "  \"fit_cpu_s\": " << fit_cpu_s << ",\n";
  out << "  \"mu_cb\": " << mu_cb.getVal() << ",\n";
  out << "  \"mu_cb_err\": " << mu_cb.getError() << ",\n";
  out << "  \"sigma_cb\": " << sigma_cb.getVal() << ",\n";
  out << "  \"sigma_cb_err\": " << sigma_cb.getError() << ",\n";
  out << "  \"lambda_bkg\": " << lambda_bkg.getVal() << ",\n";
  out << "  \"lambda_bkg_err\": " << lambda_bkg.getError() << "\n";
  out << "}\n";
  out.close();
}
"""
    roofit_macro = _render_tokens(
        roofit_macro_tmpl,
        {
            "__LO__": obs_bounds[0],
            "__HI__": obs_bounds[1],
            "__ALPHA__": alpha_cb,
            "__N_TAIL__": n_cb,
            "__N_SIG__": n_sig,
            "__N_BKG__": n_bkg,
            "__N_TOTAL__": int(n_events * 2),
            "__N_SIG_F__": float(n_sig),
        },
    )

    return {
        "kind": "cb_exp_ext",
        "observables": [{"name": "mass", "bounds": list(obs_bounds)}],
        "columns": {"mass": data},
        "ns_spec": ns_spec,
        "roofit": {"fn": "roofit_cb_exp", "macro": roofit_macro},
        "zfit": {
            "model_kind": "cb_exp_ext",
            "bounds": obs_bounds,
            "data": data,
            "init": {
                "mu_cb": 91.0,
                "sigma_cb": 3.0,
                "lambda_bkg": -0.02,
                "nsig": float(n_sig),
                "nbkg": float(n_bkg),
                "alpha_cb": float(alpha_cb),
                "n_cb": float(n_cb),
            },
            "limits": {
                "mu_cb": (85.0, 95.0),
                "sigma_cb": (0.5, 10.0),
                "lambda_bkg": (-0.1, -0.001),
                "nsig": (0.0, float(n_events * 2)),
                "nbkg": (float(n_bkg), float(n_bkg)),
                "alpha_cb": (float(alpha_cb), float(alpha_cb)),
                "n_cb": (float(n_cb), float(n_cb)),
            },
            "constants": {"nbkg": float(n_bkg), "alpha_cb": float(alpha_cb), "n_cb": float(n_cb)},
        },
    }


def _case_product2d(rng: np.random.Generator, n_events: int) -> Dict[str, Any]:
    xb = (60.0, 120.0)
    yb = (0.0, 40.0)
    truth = {"x_mu": 91.2, "x_sigma": 2.5, "y_lambda": -0.08}

    x = _sample_truncated_gaussian(rng, truth["x_mu"], truth["x_sigma"], *xb, n_events)
    y = _sample_truncated_exponential(rng, truth["y_lambda"], *yb, n_events)

    ns_spec = {
        "schema_version": "nextstat_unbinned_spec_v0",
        "model": {
            "parameters": [
                {"name": "x_mu", "init": 91.0, "bounds": [85.0, 95.0]},
                {"name": "x_sigma", "init": 2.5, "bounds": [0.5, 10.0]},
                {"name": "y_lambda", "init": -0.08, "bounds": [-1.0, -1e-4]},
            ]
        },
        "channels": [
            {
                "name": "SR",
                "data": {"file": "__DATA_PATH__"},
                "observables": [
                    {"name": "x", "bounds": list(xb)},
                    {"name": "y", "bounds": list(yb)},
                ],
                "processes": [
                    {
                        "name": "p",
                        "pdf": {
                            "type": "product",
                            "components": [
                                {"type": "gaussian", "observable": "x", "params": ["x_mu", "x_sigma"]},
                                {"type": "exponential", "observable": "y", "params": ["y_lambda"]},
                            ],
                        },
                        "yield": {"type": "fixed", "value": float(n_events)},
                    }
                ],
            }
        ],
    }

    roofit_macro_tmpl = r"""
#include <RooRealVar.h>
#include <RooGaussian.h>
#include <RooExponential.h>
#include <RooProdPdf.h>
#include <RooExtendPdf.h>
#include <RooDataSet.h>
#include <RooMinimizer.h>
#include <TStopwatch.h>
#include <memory>
#include <fstream>
#include <iomanip>

void roofit_product2d() {
  RooRealVar x("x", "x", __XLO__, __XHI__);
  RooRealVar y("y", "y", __YLO__, __YHI__);
  RooDataSet ds("ds", "ds", RooArgSet(x, y));
  {
    std::ifstream infile("__DATA_TXT__");
    double xv, yv;
    while (infile >> xv >> yv) {
      x.setVal(xv);
      y.setVal(yv);
      ds.add(RooArgSet(x, y));
    }
  }

  RooRealVar x_mu("x_mu", "x_mu", 91.0, 85.0, 95.0);
  RooRealVar x_sigma("x_sigma", "x_sigma", 2.5, 0.5, 10.0);
  RooGaussian gx("gx", "gx", x, x_mu, x_sigma);

  RooRealVar y_lambda("y_lambda", "y_lambda", -0.08, -1.0, -1e-4);
  RooExponential ey("ey", "ey", y, y_lambda);

  RooProdPdf model("model", "model", RooArgList(gx, ey));
  RooRealVar nobs("nobs", "nobs", __NOBS__, 0, __NOBS__ * 2);
  nobs.setConstant(true);
  RooExtendPdf model_ext("model_ext", "model_ext", model, nobs);

  TStopwatch sw;
  sw.Start();
  std::unique_ptr<RooAbsReal> nll(model_ext.createNLL(ds, RooFit::Extended(true)));
  RooMinimizer m(*nll);
  m.setPrintLevel(-1);
  m.setEps(1e-12);
  m.setStrategy(1);
  int status = m.minimize("Minuit2", "Migrad");
  if (status != 0) {
    m.setStrategy(2);
    status = m.minimize("Minuit2", "Migrad");
  }
  m.hesse();
  sw.Stop();

  double nll_val = nll->getVal();
  double fit_real_s = sw.RealTime();
  double fit_cpu_s = sw.CpuTime();

  std::ofstream out("roofit_result.json");
  out << std::setprecision(17);
  out << "{\n";
  out << "  \"tool\": \"roofit\",\n";
  out << "  \"status\": " << status << ",\n";
  out << "  \"nll\": " << nll_val << ",\n";
  out << "  \"fit_real_s\": " << fit_real_s << ",\n";
  out << "  \"fit_cpu_s\": " << fit_cpu_s << ",\n";
  out << "  \"x_mu\": " << x_mu.getVal() << ",\n";
  out << "  \"x_mu_err\": " << x_mu.getError() << ",\n";
  out << "  \"x_sigma\": " << x_sigma.getVal() << ",\n";
  out << "  \"x_sigma_err\": " << x_sigma.getError() << ",\n";
  out << "  \"y_lambda\": " << y_lambda.getVal() << ",\n";
  out << "  \"y_lambda_err\": " << y_lambda.getError() << "\n";
  out << "}\n";
  out.close();
}
"""
    roofit_macro = _render_tokens(
        roofit_macro_tmpl,
        {
            "__XLO__": xb[0],
            "__XHI__": xb[1],
            "__YLO__": yb[0],
            "__YHI__": yb[1],
            "__NOBS__": int(n_events),
        },
    )

    data2 = np.column_stack([x, y]).astype(np.float64)
    return {
        "kind": "product2d",
        "observables": [{"name": "x", "bounds": list(xb)}, {"name": "y", "bounds": list(yb)}],
        "columns": {"x": x, "y": y},
        "ns_spec": ns_spec,
        "roofit": {"fn": "roofit_product2d", "macro": roofit_macro},
        "zfit": {
            "model_kind": "product2d",
            "bounds": (xb, yb),
            "data": data2,
            "init": {"x_mu": 91.0, "x_sigma": 2.5, "y_lambda": -0.08},
            "limits": {"x_mu": (85.0, 95.0), "x_sigma": (0.5, 10.0), "y_lambda": (-1.0, -1e-4)},
            "constants": {},
        },
    }


CASE_BUILDERS = {
    "gauss_exp": _case_gauss_exp,
    "cb_exp": _case_cb_exp,
    "product2d": _case_product2d,
}


def _write_data_txt_1d(values: np.ndarray, path: Path) -> None:
    with path.open("w") as f:
        for v in values:
            f.write(f"{float(v):.17g}\n")


def _write_data_txt_2d(values: np.ndarray, path: Path) -> None:
    with path.open("w") as f:
        for row in values:
            f.write(f"{float(row[0]):.17g} {float(row[1]):.17g}\n")


def main() -> None:
    ap = argparse.ArgumentParser(description="Cross-framework unbinned benchmark suite")
    ap.add_argument("--cases", default="all", help="Comma-separated: gauss_exp,cb_exp,product2d or 'all'")
    ap.add_argument("--n-events", type=int, default=100_000, help="Number of events per case (toy generation)")
    ap.add_argument("--seed", type=int, default=42, help="RNG seed")
    ap.add_argument("--out", type=str, default=None, help="Output JSON path (default: print to stdout)")
    ap.add_argument("--keep", action="store_true", help="Keep temporary work directory")
    args = ap.parse_args()

    keys = sorted(CASE_BUILDERS.keys()) if args.cases == "all" else [k.strip() for k in args.cases.split(",")]
    for k in keys:
        if k not in CASE_BUILDERS:
            _die(f"Unknown case '{k}'. Available: {sorted(CASE_BUILDERS)}")

    rng = np.random.default_rng(args.seed)
    workdir = Path(tempfile.mkdtemp(prefix="ns_unbinned_bench_"))

    suite: Dict[str, Any] = {
        "schema_version": "nextstat.unbinned_run_suite_result.v1",
        "suite": "benchmarks/unbinned/run_suite.py",
        "seed": args.seed,
        "n_events": args.n_events,
        "cases": [],
        "availability": {
            "root": _has_root(),
            "zfit": _has_import("zfit"),
            "morefit": bool(os.environ.get("MOREFIT_BIN") or shutil.which("morefit_gauss_exp")),
        },
    }

    try:
        for k in keys:
            case_dir = workdir / k
            case_dir.mkdir(parents=True, exist_ok=True)

            built = CASE_BUILDERS[k](rng, args.n_events)

            # Data
            parquet_path = case_dir / "observed.parquet"
            _write_parquet(columns=built["columns"], observables=built["observables"], path=parquet_path)

            # NextStat spec
            spec = built["ns_spec"]
            spec["channels"][0]["data"]["file"] = str(parquet_path)
            spec_path = case_dir / "spec.json"
            spec_path.write_text(json.dumps(spec, indent=2))

            # NextStat fit
            ns_result: Dict[str, Any]
            try:
                ns_result = _fit_nextstat(spec_path)
            except Exception as e:
                ns_result = {"failed": True, "reason": str(e)}

            # RooFit fit (optional)
            roofit_result: Dict[str, Any] = {"skipped": True, "reason": "root not found on PATH"}
            if _has_root():
                data_txt = case_dir / "observed.txt"
                if k == "product2d":
                    data2 = np.column_stack([built["columns"]["x"], built["columns"]["y"]])
                    _write_data_txt_2d(data2, data_txt)
                else:
                    _write_data_txt_1d(built["columns"]["mass"], data_txt)

                macro_path = case_dir / f"{built['roofit']['fn']}.C"
                macro_text = built["roofit"]["macro"].replace("__DATA_TXT__", str(data_txt))
                macro_path.write_text(macro_text)
                roofit_result = _run_root_macro(macro_path, built["roofit"]["fn"])

            # zfit fit (optional; runs in a subprocess with timeout)
            zfit_result = {"skipped": True, "reason": "zfit not importable"}
            try:
                zcfg = built["zfit"]
                zfit_result = _fit_zfit_with_timeout(
                    case_name=k,
                    data=zcfg["data"],
                    bounds=zcfg["bounds"],
                    model_kind=zcfg["model_kind"],
                    init=zcfg["init"],
                    limits=zcfg["limits"],
                    constants=zcfg["constants"],
                )
            except Exception as e:
                zfit_result = {"failed": True, "reason": f"zfit runner error: {e}"}

            # morefit fit (optional; C++ binary via subprocess)
            try:
                morefit_data = built["columns"].get("mass", np.array([]))
                morefit_bounds = built.get("observables", [{}])[0].get("bounds", (60.0, 120.0))
                if isinstance(morefit_bounds, list):
                    morefit_bounds = tuple(morefit_bounds)
                morefit_result = _fit_morefit(
                    case_name=k,
                    data=morefit_data,
                    bounds=morefit_bounds,
                    model_kind=built["kind"],
                    tmpdir=case_dir,
                )
            except Exception as e:
                morefit_result = {"failed": True, "reason": f"morefit runner error: {e}"}

            suite["cases"].append(
                {
                    "case": k,
                    "kind": built["kind"],
                    "nextstat": ns_result,
                    "roofit": roofit_result,
                    "zfit": zfit_result,
                    "morefit": morefit_result,
                }
            )

    finally:
        if args.keep:
            suite["workdir"] = str(workdir)
        else:
            shutil.rmtree(workdir, ignore_errors=True)

    out_json = json.dumps(suite, indent=2, sort_keys=True)
    if args.out:
        out_path = Path(args.out)
        out_path.parent.mkdir(parents=True, exist_ok=True)
        out_path.write_text(out_json + "\n")
    else:
        print(out_json)


if __name__ == "__main__":
    main()
