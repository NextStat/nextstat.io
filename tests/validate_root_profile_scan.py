#!/usr/bin/env python3
"""Validate NextStat profile-scan vs ROOT HistFactory reference.

Supports two input modes:

1) From pyhf JSON (NextStat-native):
   - Exports to HistFactory XML + ROOT histograms via `pyhf.writexml` (requires `uproot`).
   - Builds a RooWorkspace via `hist2workspace` (ROOT).

2) From an existing HistFactory Combination XML (e.g. exported by TRExFitter):
   - Parses to a pyhf workspace via `pyhf.readxml` (requires `uproot`).
   - Builds a RooWorkspace via `hist2workspace` (ROOT).

Then:
  - ROOT: performs a free fit and a fixed-POI scan to produce q(mu).
  - NextStat: runs `nextstat.infer.profile_scan` on the same mu grid.
  - Compares q(mu), mu_hat, and NLL values.

Run (example):
  PYTHONPATH=bindings/ns-py/python ./.venv/bin/python tests/validate_root_profile_scan.py \\
    --pyhf-json tests/fixtures/simple_workspace.json \\
    --measurement GaussExample \\
    --start 0.0 --stop 5.0 --points 51
"""

from __future__ import annotations

import argparse
import json
import os
import shutil
import subprocess
import sys
import time
from pathlib import Path
from typing import Any, Dict, List, Optional, Tuple

try:
    import pyhf  # type: ignore
except ModuleNotFoundError:
    pyhf = None  # type: ignore

import nextstat
import nextstat.infer as ns_infer


def _which(name: str) -> Optional[str]:
    return shutil.which(name)


def _run(cmd: List[str], *, cwd: Path) -> None:
    p = subprocess.run(cmd, cwd=str(cwd), stdout=subprocess.PIPE, stderr=subprocess.STDOUT, text=True)
    if p.returncode != 0:
        raise RuntimeError(f"Command failed ({p.returncode}): {' '.join(cmd)}\n{p.stdout}")


def _load_pyhf_json(path: Path) -> Dict[str, Any]:
    return json.loads(path.read_text())


def _parse_histfactory_to_pyhf_workspace(combination_xml: Path, rootdir: Path) -> Dict[str, Any]:
    if pyhf is None:
        raise RuntimeError(
            "pyhf is required for HistFactory XML import. Install `pyhf[xmlio]` (needs uproot)."
        )
    try:
        # NOTE: avoid `import pyhf.readxml` here because it binds a local `pyhf` name
        # and can trigger UnboundLocalError when we also reference the module-global `pyhf`.
        from pyhf import readxml as readxml  # type: ignore
    except ModuleNotFoundError as e:
        raise RuntimeError(
            "pyhf.readxml requires `uproot`. Install it (e.g. `pip install uproot`) "
            "or install the project extras that include it."
        ) from e

    return readxml.parse(str(combination_xml), rootdir=str(rootdir), track_progress=False)


def _export_pyhf_to_histfactory(ws: Dict[str, Any], out_dir: Path, *, prefix: str) -> Path:
    if pyhf is None:
        raise RuntimeError(
            "pyhf is required for pyhf->HistFactory export. Install `pyhf[xmlio]` (needs uproot)."
        )
    try:
        # NOTE: avoid `import pyhf.writexml` here because it binds a local `pyhf` name
        # and can trigger UnboundLocalError when we also reference the module-global `pyhf`.
        from pyhf import writexml as writexml  # type: ignore
    except ModuleNotFoundError as e:
        raise RuntimeError(
            "pyhf.writexml requires `uproot`. Install it (e.g. `pip install uproot`) "
            "or install the project extras that include it."
        ) from e

    out_dir.mkdir(parents=True, exist_ok=True)
    spec_dir = out_dir / "spec"
    spec_dir.mkdir(parents=True, exist_ok=True)

    # writexml copies the DTD to `spec_dir.parent` and writes channel XMLs into spec_dir.
    combo_bytes = writexml.writexml(ws, specdir=str(spec_dir), data_rootdir=str(out_dir), resultprefix=prefix)
    combo_path = out_dir / "combination.xml"
    combo_path.write_bytes(combo_bytes)
    return combo_path


def _hist2workspace(combination_xml: Path) -> Path:
    hist2ws = _which("hist2workspace")
    if not hist2ws:
        raise RuntimeError(
            "ROOT `hist2workspace` not found in PATH. "
            "Make sure ROOT is installed and your environment is set up (thisroot.sh)."
        )

    workdir = combination_xml.parent
    before = {p.resolve() for p in workdir.rglob("*.root")}
    # `hist2workspace` expects the DTD relative to the XML; run in the same directory.
    _run([hist2ws, combination_xml.name], cwd=workdir)
    after = {p.resolve() for p in workdir.rglob("*.root")}

    new_root_files = sorted((after - before), key=lambda p: p.stat().st_mtime, reverse=True)
    new_root_files = [p for p in new_root_files if p.name != "data.root"]
    if not new_root_files:
        raise RuntimeError(
            "hist2workspace did not create a new ROOT workspace file (only data.root found). "
            "Check hist2workspace output and the Combination XML OutputFilePrefix."
        )
    return new_root_files[0]

def _stage_histfactory_export(
    *,
    combination_xml: Path,
    rootdir: Path,
    run_dir: Path,
) -> Path:
    """Stage a HistFactory export directory into `run_dir`.

    `hist2workspace` resolves paths in `combination.xml` relative to the current working
    directory (and expects the DTD and channel XMLs to be available relative to it).
    Many exports (including those created by `pyhf.writexml`) use relative paths.

    To make runs reproducible and independent of the source location, we copy the
    export directory contents into `run_dir` and run `hist2workspace` there.
    """
    rootdir = rootdir.resolve()
    run_dir = run_dir.resolve()

    if not rootdir.exists() or not rootdir.is_dir():
        raise RuntimeError(f"rootdir does not exist or is not a directory: {rootdir}")

    # Best-effort: copy top-level files/dirs (channels/, data.root, DTD, etc.).
    for p in sorted(rootdir.iterdir(), key=lambda x: x.name):
        dest = run_dir / p.name
        if p.is_dir():
            # Merge into an existing destination if re-running with --keep.
            shutil.copytree(p, dest, dirs_exist_ok=True)
        elif p.is_file():
            # Avoid copying previously generated hist2workspace outputs which would
            # break our before/after detection logic (and can confuse ROOT).
            if p.suffix == ".root" and p.name.endswith("_model.root"):
                continue
            shutil.copy2(p, dest)

    staged_combo = run_dir / "combination.xml"
    if combination_xml.resolve() != staged_combo:
        shutil.copy2(combination_xml, staged_combo)
    return staged_combo


def _write_root_macro_profile_scan(
    macro_path: Path,
    *,
    root_workspace_file: Path,
    mu_values: List[float],
    out_json: Path,
    dump_params: bool,
) -> None:
    mu_list = ", ".join(f"{x:.16g}" for x in mu_values)
    params_hat_code = ""
    params_mu_decl_code = ""
    params_mu_push_code = ""
    json_header_code = ""
    point_params_json_code = ""
    if dump_params:
        params_hat_code = r"""
  std::vector<double> params_hat;
  params_hat.reserve(params.size());
  for (auto* p : params) params_hat.push_back(p->getVal());
"""
        params_mu_decl_code = r"""
  std::vector<std::vector<double>> params_mu;
  params_mu.reserve(mu_values.size());
"""
        params_mu_push_code = r"""
    std::vector<double> pv;
    pv.reserve(params.size());
    for (auto* p : params) pv.push_back(p->getVal());
    params_mu.push_back(pv);
"""
        json_header_code = r"""
  out << "  \"parameter_names\": [";
  for (size_t i = 0; i < params.size(); ++i) {
    out << "\"" << params[i]->GetName() << "\"";
    if (i + 1 != params.size()) out << ", ";
  }
  out << "],\n";
  out << "  \"params_hat\": [";
  for (size_t i = 0; i < params_hat.size(); ++i) {
    out << params_hat[i];
    if (i + 1 != params_hat.size()) out << ", ";
  }
  out << "],\n";
"""
        point_params_json_code = r"""
    out << ", \"params\": [";
    for (size_t j = 0; j < params_mu[i].size(); ++j) {
      out << params_mu[i][j];
      if (j + 1 != params_mu[i].size()) out << ", ";
    }
    out << "]";
"""
    macro = f"""
#include <TFile.h>
#include <TKey.h>
#include <TSystem.h>
#include <TStopwatch.h>

#include <RooWorkspace.h>
#include <RooAbsPdf.h>
#include <RooAbsData.h>
#include <RooAbsReal.h>
#include <RooAbsCollection.h>
#include <RooRealVar.h>
#include <RooArgSet.h>
#include <RooMinimizer.h>

#include <RooStats/ModelConfig.h>

#include <algorithm>
#include <cmath>
#include <fstream>
#include <iomanip>
#include <string>
#include <vector>

static RooWorkspace* find_workspace(TFile& f) {{
  RooWorkspace* w = nullptr;
  TIter next(f.GetListOfKeys());
  while (auto* k = (TKey*)next()) {{
    auto cname = std::string(k->GetClassName());
    if (cname.find("RooWorkspace") == std::string::npos) continue;
    w = (RooWorkspace*)k->ReadObj();
    if (w) return w;
  }}
  return nullptr;
}}

static RooStats::ModelConfig* find_model_config(RooWorkspace& w) {{
  // Standard HistFactory name.
  if (auto* o = w.obj("ModelConfig")) {{
    return dynamic_cast<RooStats::ModelConfig*>(o);
  }}
  if (auto* o = w.obj("modelConfig")) {{
    return dynamic_cast<RooStats::ModelConfig*>(o);
  }}
  return nullptr;
}}

static RooAbsData* find_data(RooWorkspace& w) {{
  if (auto* d = w.data("obsData")) return d;
  if (auto* d = w.data("data")) return d;
  // HistFactory workspaces should provide "obsData". Avoid version-dependent
  // RooWorkspace::allData() container APIs (ROOT 6.38 changed this surface).
  return nullptr;
}}

static int minimize_nll(RooAbsReal& nll) {{
  RooMinimizer m(nll);
  m.setPrintLevel(-1);
  m.setEps(1e-12);
  m.setMaxFunctionCalls(200000);
  m.setMaxIterations(200000);
  m.setOffsetting(true);
  m.optimizeConst(2);
  // Strategy 0 is fast but can fail on some realistic exports; keep a retry ladder.
  m.setStrategy(1);
  int status = m.minimize("Minuit2", "Migrad");
  if (status != 0) {{
    m.setStrategy(2);
    status = m.minimize("Minuit2", "Migrad");
  }}
  if (status != 0) {{
    (void)m.minimize("Minuit2", "Simplex");
    status = m.minimize("Minuit2", "Migrad");
  }}
  return status;
}}

void profile_scan() {{
  gSystem->Load("libRooFit");
  gSystem->Load("libRooStats");

  const char* root_path = "{root_workspace_file.as_posix()}";
  const char* out_path = "{out_json.as_posix()}";

  std::vector<double> mu_values = {{ {mu_list} }};

  TStopwatch sw_total;
  sw_total.Start();

  TFile f(root_path, "READ");
  if (f.IsZombie()) {{
    throw std::runtime_error(std::string("Failed to open ROOT file: ") + root_path);
  }}

  RooWorkspace* w = find_workspace(f);
  if (!w) {{
    throw std::runtime_error("No RooWorkspace found in file.");
  }}

  auto* mc = find_model_config(*w);
  if (!mc) {{
    throw std::runtime_error("ModelConfig not found (expected 'ModelConfig' in workspace).");
  }}

  RooAbsPdf* pdf = mc->GetPdf();
  if (!pdf) {{
    throw std::runtime_error("ModelConfig has no PDF.");
  }}

  RooAbsData* data = find_data(*w);
  if (!data) {{
    throw std::runtime_error("Observed dataset not found (expected 'obsData').");
  }}

  // Assume a single POI.
  auto* poi_set = mc->GetParametersOfInterest();
  if (!poi_set || poi_set->getSize() == 0) {{
    throw std::runtime_error("POI set is empty.");
  }}
  auto* poi = dynamic_cast<RooRealVar*>(poi_set->first());
  if (!poi) {{
    throw std::runtime_error("POI is not a RooRealVar.");
  }}

  // Build NLL (HistFactory convention).
  RooArgSet empty;
  const RooArgSet* nuis = mc->GetNuisanceParameters();
  const RooArgSet* globs = mc->GetGlobalObservables();
  if (!nuis) nuis = &empty;
  if (!globs) globs = &empty;

  // Optional: collect a stable parameter ordering for debugging.
  // We use pdf->getParameters(data) to capture any RooFit-side parameters beyond the
  // explicit ModelConfig POI/nuisance sets (e.g. lumi factors), then sort by name.
  std::unique_ptr<RooArgSet> params_set(pdf->getParameters(*data));
  if (!params_set) params_set.reset(new RooArgSet());
  std::vector<RooRealVar*> params;
  params.reserve((size_t)params_set->getSize());
  for (auto* a : *params_set) {{
    if (auto* v = dynamic_cast<RooRealVar*>(a)) params.push_back(v);
  }}
  std::sort(params.begin(), params.end(), [](const RooRealVar* a, const RooRealVar* b) {{
    return std::string(a->GetName()) < std::string(b->GetName());
  }});
  params.erase(std::unique(params.begin(), params.end(), [](const RooRealVar* a, const RooRealVar* b) {{
    return std::string(a->GetName()) == std::string(b->GetName());
  }}), params.end());
  std::unique_ptr<RooAbsReal> nll(pdf->createNLL(
      *data,
      RooFit::Extended(true),
      RooFit::Constrain(*nuis),
      RooFit::GlobalObservables(*globs)
  ));

  // Free fit
  poi->setConstant(false);
  // Tighten POI range around the scan grid to avoid unbounded runaway in some exports.
  // Keep this symmetric and padded so the unconditional minimum can still be found.
  double mu_lo = *std::min_element(mu_values.begin(), mu_values.end());
  double mu_hi = *std::max_element(mu_values.begin(), mu_values.end());
  double mu_pad = (mu_hi - mu_lo) * 10.0 + 1.0;
  poi->setRange(mu_lo - mu_pad, mu_hi + mu_pad);
  poi->setVal(0.5 * (mu_lo + mu_hi));
  TStopwatch sw_free;
  sw_free.Start();
  int status_free = minimize_nll(*nll);
  sw_free.Stop();
  double mu_hat = poi->getVal();
  double nll_hat = nll->getVal();
{params_hat_code}

  // Fixed-POI scan
  std::vector<double> nll_mu;
  std::vector<int> status_mu;
{params_mu_decl_code}
  nll_mu.reserve(mu_values.size());
  status_mu.reserve(mu_values.size());

  for (double mu : mu_values) {{
    poi->setVal(mu);
    poi->setConstant(true);
    // Rebuild NLL to avoid stale caches when toggling constants.
    std::unique_ptr<RooAbsReal> nll_fixed(pdf->createNLL(
        *data,
        RooFit::Extended(true),
        RooFit::Constrain(*nuis),
        RooFit::GlobalObservables(*globs)
    ));
    int st = minimize_nll(*nll_fixed);
    double v = nll_fixed->getVal();
    nll_mu.push_back(v);
    status_mu.push_back(st);
{params_mu_push_code}
  }}
  poi->setConstant(false);

  // If unconditional fit failed, fall back to scan minimum for mu_hat/nll_hat
  // (requires the scan grid to cover the minimum reasonably well).
  bool free_fit_ok = (status_free == 0) && std::isfinite(mu_hat) && std::isfinite(nll_hat);
  const char* mu_hat_source = free_fit_ok ? "free_fit" : "scan_min";
  if (!free_fit_ok) {{
    size_t imin = 0;
    for (size_t i = 1; i < nll_mu.size(); ++i) {{
      if (nll_mu[i] < nll_mu[imin]) imin = i;
    }}
    mu_hat = mu_values[imin];
    nll_hat = nll_mu[imin];
  }}

  // Match NextStat `profile_scan` semantics: one-sided q(mu).
  std::vector<double> q_mu;
  q_mu.reserve(mu_values.size());
  for (size_t i = 0; i < mu_values.size(); ++i) {{
    double mu = mu_values[i];
    double v = nll_mu[i];
    double q = 0.0;
    if (mu >= mu_hat) {{
      q = 2.0 * (v - nll_hat);
      if (q < 0.0) q = 0.0;
    }}
    q_mu.push_back(q);
  }}

  sw_total.Stop();

  std::ofstream out(out_path);
  out << std::setprecision(17);
  out << "{{\\n";
  out << "  \\"tool\\": \\"root\\",\\n";
  out << "  \\"poi_name\\": \\"" << poi->GetName() << "\\",\\n";
  out << "  \\"mu_hat\\": " << mu_hat << ",\\n";
  out << "  \\"nll_hat\\": " << nll_hat << ",\\n";
  out << "  \\"status_free\\": " << status_free << ",\\n";
  out << "  \\"mu_hat_source\\": \\"" << mu_hat_source << "\\",\\n";
  out << "  \\"timing_s\\": {{\\"total\\": " << sw_total.RealTime() << ", \\"free_fit\\": " << sw_free.RealTime() << "}},\\n";
{json_header_code}
  out << "  \\"points\\": [\\n";
  for (size_t i = 0; i < mu_values.size(); ++i) {{
    out << "    {{\\"mu\\": " << mu_values[i]
        << ", \\"nll_mu\\": " << nll_mu[i]
        << ", \\"q_mu\\": " << q_mu[i]
        << ", \\"status\\": " << status_mu[i];
{point_params_json_code}
    out << "}}";
    if (i + 1 != mu_values.size()) out << ",";
    out << "\\n";
  }}
  out << "  ]\\n";
  out << "}}\\n";
  out.close();
}}
"""
    macro_path.write_text(macro)


def _run_root_macro(macro_path: Path, *, cwd: Path) -> None:
    root = _which("root")
    if not root:
        raise RuntimeError(
            "ROOT `root` not found in PATH. Make sure ROOT is installed and your environment is set up."
        )
    _run([root, "-l", "-b", "-q", macro_path.name], cwd=cwd)


def _mu_grid(start: float, stop: float, points: int) -> List[float]:
    if points < 2:
        return [start]
    step = (stop - start) / (points - 1)
    return [start + i * step for i in range(points)]

def _measurement_name_from_combination_xml(path: Path) -> str:
    # Minimal XML parse: pick first <Measurement Name="...">.
    try:
        import xml.etree.ElementTree as ET

        root = ET.fromstring(path.read_text())
        for meas in root.findall(".//Measurement"):
            name = meas.attrib.get("Name") or meas.attrib.get("name")
            if name:
                return str(name)
    except Exception:
        pass
    return "measurement"

def _pyhf_profile_scan(
    ws_spec: Dict[str, Any],
    *,
    measurement_name: str,
    mu_values: List[float],
) -> Dict[str, Any]:
    if pyhf is None:
        raise RuntimeError("pyhf is required for --include-pyhf")
    from pyhf.infer import mle, test_statistics

    ws = pyhf.Workspace(ws_spec)
    model = ws.model(measurement_name)
    data = ws.data(model, measurement_name)

    init = model.config.suggested_init()
    bounds = model.config.suggested_bounds()
    fixed = model.config.suggested_fixed()

    fit_pars, twice_nll_hat = mle.fit(
        data,
        model,
        init_pars=init,
        par_bounds=bounds,
        fixed_params=fixed,
        return_fitted_val=True,
    )
    poi_idx = model.config.poi_index
    mu_hat = float(fit_pars[poi_idx])

    points = []
    for mu in mu_values:
        # If the POI is bounded at 0 (common in HistFactory/HEP), pyhf recommends qmu_tilde.
        # For mu_hat >= 0 (the usual case here) this matches qmu but avoids noisy warnings.
        qmu = float(test_statistics.qmu_tilde(mu, data, model, init, bounds, fixed))
        points.append({"mu": float(mu), "q_mu": qmu})

    return {
        "tool": "pyhf",
        "poi_name": str(model.config.poi_name),
        "mu_hat": mu_hat,
        "twice_nll_hat": float(twice_nll_hat),
        "points": points,
    }


def main() -> int:
    ap = argparse.ArgumentParser()
    src = ap.add_mutually_exclusive_group(required=True)
    src.add_argument("--pyhf-json", type=Path, help="Path to a pyhf workspace JSON.")
    src.add_argument(
        "--histfactory-xml",
        type=Path,
        help="Path to HistFactory Combination XML (e.g. produced by TRExFitter).",
    )
    ap.add_argument(
        "--rootdir",
        type=Path,
        default=None,
        help="Root directory for resolving relative paths in HistFactory XML (default: XML parent).",
    )
    ap.add_argument("--measurement", type=str, default=None, help="pyhf measurement name (for --pyhf-json).")
    ap.add_argument("--start", type=float, default=0.0)
    ap.add_argument("--stop", type=float, default=5.0)
    ap.add_argument("--points", type=int, default=51)
    ap.add_argument("--workdir", type=Path, default=Path("tmp/root_parity"))
    ap.add_argument("--keep", action="store_true", help="Keep intermediate HistFactory/ROOT artifacts.")
    ap.add_argument(
        "--dump-root-params",
        action="store_true",
        help="Include ROOT fitted parameter vectors in root_profile_scan.json (diagnostic; large output).",
    )
    ap.add_argument(
        "--include-pyhf",
        action="store_true",
        help="Also compute a pyhf q(mu) scan for diagnosis (slow but canonical for pyhf semantics).",
    )
    args = ap.parse_args()

    mu_values = _mu_grid(args.start, args.stop, args.points)

    t0 = time.perf_counter()
    workdir = args.workdir.resolve()
    run_dir = workdir / f"run_{int(time.time())}"
    run_dir.mkdir(parents=True, exist_ok=True)

    # ---------------------------------------------------------------------
    # Build HistFactory artifacts (and optional pyhf workspace)
    # ---------------------------------------------------------------------
    if args.pyhf_json:
        if args.measurement is None:
            raise SystemExit("--measurement is required with --pyhf-json")
        ws = _load_pyhf_json(args.pyhf_json)
        combo_xml = _export_pyhf_to_histfactory(ws, run_dir, prefix="ns")
        rootdir = combo_xml.parent
        measurement_name = args.measurement
    else:
        combo_xml = args.histfactory_xml.resolve()
        rootdir = (args.rootdir or combo_xml.parent).resolve()
        ws = None
        measurement_name = _measurement_name_from_combination_xml(combo_xml)

    t_build_ws = time.perf_counter() - t0

    # ---------------------------------------------------------------------
    # ROOT reference: combination.xml -> RooWorkspace -> profile scan
    # ---------------------------------------------------------------------
    t1 = time.perf_counter()
    # Stage HistFactory export into run_dir so `hist2workspace` can resolve relative paths.
    if args.pyhf_json:
        # Already exported into `run_dir` via `_export_pyhf_to_histfactory`.
        combo_xml = (run_dir / "combination.xml").resolve()
    else:
        combo_xml = _stage_histfactory_export(
            combination_xml=combo_xml,
            rootdir=rootdir,
            run_dir=run_dir,
        )
    root_ws_file = _hist2workspace(combo_xml)
    t_hist2ws = time.perf_counter() - t1

    # ---------------------------------------------------------------------
    # Optional: pyhf reference scan (diagnostic)
    # ---------------------------------------------------------------------
    pyhf_scan = None
    t_pyhf = None
    pyhf_out = None
    if args.include_pyhf:
        if ws is None:
            # Parse from staged export dir so relative paths resolve correctly.
            ws = _parse_histfactory_to_pyhf_workspace(combo_xml, run_dir)
        t_py0 = time.perf_counter()
        pyhf_scan = _pyhf_profile_scan(ws, measurement_name=measurement_name, mu_values=mu_values)
        t_pyhf = time.perf_counter() - t_py0
        pyhf_out = run_dir / "pyhf_profile_scan.json"
        pyhf_out.write_text(json.dumps(pyhf_scan, indent=2))

    t2 = time.perf_counter()
    root_out = run_dir / "root_profile_scan.json"
    macro_path = run_dir / "profile_scan.C"
    _write_root_macro_profile_scan(
        macro_path,
        root_workspace_file=root_ws_file,
        mu_values=mu_values,
        out_json=root_out,
        dump_params=args.dump_root_params,
    )
    _run_root_macro(macro_path, cwd=run_dir)
    root_result = json.loads(root_out.read_text())
    t_root = time.perf_counter() - t2

    # ---------------------------------------------------------------------
    # NextStat: profile scan (direct HistFactory import, no pyhf dependency)
    # ---------------------------------------------------------------------
    t3 = time.perf_counter()
    # Validation runs should prefer numerical robustness and determinism.
    #
    # - Parity mode enables Kahan summation and other "numbers-first" behaviors.
    # - Threads=1 avoids non-deterministic reductions.
    try:
        nextstat.set_eval_mode("parity")
    except Exception:
        pass
    try:
        nextstat.set_threads(1)
    except Exception:
        pass
    ns_model = nextstat.from_histfactory_xml(str(combo_xml))
    ns_scan = ns_infer.profile_scan(ns_model, mu_values, return_params=bool(args.dump_root_params))
    t_ns = time.perf_counter() - t3
    ns_out = run_dir / "nextstat_profile_scan.json"
    ns_out.write_text(json.dumps(ns_scan, indent=2))

    # Compute q(mu) diffs early so diagnostics can focus on the scan-mismatch point.
    root_points = list(root_result.get("points") or [])
    ns_points = list(ns_scan.get("points") or [])
    if len(root_points) != len(mu_values) or len(ns_points) != len(mu_values):
        raise RuntimeError(
            f"unexpected point count: root={len(root_points)} nextstat={len(ns_points)} expected={len(mu_values)}"
        )

    def _nearest_point(points: list[dict[str, Any]], mu: float) -> dict[str, Any]:
        best = min(points, key=lambda p: abs(float(p.get("mu")) - float(mu)))
        if abs(float(best.get("mu")) - float(mu)) > 1e-8:
            raise RuntimeError(f"no matching mu point found: mu={mu} best={best.get('mu')}")
        return best

    q_diffs: List[Tuple[float, float]] = []
    for i, mu_expected in enumerate(mu_values):
        p_root = root_points[i]
        p_ns = ns_points[i]
        mu_root = float(p_root.get("mu"))
        mu_ns = float(p_ns.get("mu"))

        # Defensive: if ordering drifts for any reason, fall back to nearest matching point.
        if abs(mu_root - float(mu_expected)) > 1e-8:
            p_root = _nearest_point(root_points, float(mu_expected))
        if abs(mu_ns - float(mu_expected)) > 1e-8:
            p_ns = _nearest_point(ns_points, float(mu_expected))

        q_root = float(p_root.get("q_mu"))
        q_ns = float(p_ns.get("q_mu"))
        q_diffs.append((float(mu_expected), q_ns - q_root))

    dq_max_mu: Optional[float] = None
    if q_diffs:
        dq_max_mu = max(q_diffs, key=lambda t: abs(t[1]))[0]

    diagnostic: Dict[str, Any] = {}
    if args.dump_root_params and "parameter_names" in root_result and "params_hat" in root_result:
        try:
            root_names = list(root_result["parameter_names"])
            ns_names = list(ns_model.parameter_names())
            if not ns_names:
                raise RuntimeError("NextStat model did not expose parameter_names()")

            def _ns_params_from_root_map(m: Dict[str, float]) -> List[float]:
                out = []
                for n in ns_names:
                    if n in m:
                        out.append(float(m[n]))
                        continue
                    # RooFit/HistFactory naming conventions:
                    # - nuisance parameters: alpha_<name>
                    # - norm factors: NF_<name>
                    alpha = f"alpha_{n}"
                    nf = f"NF_{n}"
                    if alpha in m:
                        out.append(float(m[alpha]))
                        continue
                    if nf in m:
                        out.append(float(m[nf]))
                        continue
                    if n.startswith("staterror_") and "[" in n and n.endswith("]"):
                        # NextStat staterror naming: staterror_<channel>[<bin>]
                        # RooFit/HistFactory naming: gamma_stat_<channel>_bin_<bin>
                        try:
                            tail = n[len("staterror_") :]
                            chan, idx_part = tail.rsplit("[", 1)
                            idx = idx_part[:-1]
                            gamma = f"gamma_stat_{chan}_bin_{idx}"
                            if gamma in m:
                                out.append(float(m[gamma]))
                                continue
                        except Exception:
                            pass
                    if n.startswith("shape_stat_") and "[" in n and n.endswith("]"):
                        # NextStat shapesys-stat naming: shape_stat_<...>[<bin>]
                        # RooFit/HistFactory naming: gamma_shape_stat_<...>_bin_<bin>
                        try:
                            tail = n[len("shape_stat_") :]
                            base, idx_part = tail.rsplit("[", 1)
                            idx = idx_part[:-1]
                            gamma = f"gamma_shape_stat_{base}_bin_{idx}"
                            if gamma in m:
                                out.append(float(m[gamma]))
                                continue
                        except Exception:
                            pass
                    raise KeyError(n)
                return out

            root_hat = dict(zip(root_names, map(float, root_result["params_hat"])))
            ns_nll_hat_at_root = float(ns_model.nll(_ns_params_from_root_map(root_hat)))
            diagnostic["ns_nll_at_root_hat_params"] = ns_nll_hat_at_root
            diagnostic["delta_nll_hat_ns_minus_root"] = ns_nll_hat_at_root - float(root_result["nll_hat"])

            deltas: List[Tuple[float, float]] = []
            for p in root_result.get("points") or []:
                if "params" not in p:
                    continue
                mu = float(p["mu"])
                root_map = dict(zip(root_names, map(float, p["params"])))
                ns_nll = float(ns_model.nll(_ns_params_from_root_map(root_map)))
                delta = ns_nll - float(p["nll_mu"])
                deltas.append((mu, delta))

            if deltas:
                diagnostic["delta_nll_mu_ns_minus_root_min"] = min(d for _, d in deltas)
                diagnostic["delta_nll_mu_ns_minus_root_max"] = max(d for _, d in deltas)
                diagnostic["delta_nll_mu_ns_minus_root_span"] = (
                    diagnostic["delta_nll_mu_ns_minus_root_max"]
                    - diagnostic["delta_nll_mu_ns_minus_root_min"]
                )

            poi_idx = int(ns_scan.get("poi_index") or 0)

            def _grad_stats(params: List[float]) -> Dict[str, float]:
                g = list(map(float, ns_model.grad_nll(params)))
                # For fixed-mu fits, POI is clamped and its gradient is not expected to be 0.
                if 0 <= poi_idx < len(g):
                    g[poi_idx] = 0.0
                maxabs = max(abs(x) for x in g) if g else 0.0
                l2 = float(sum(x * x for x in g) ** 0.5)
                return {"grad_l2": l2, "grad_max_abs": float(maxabs)}

            # Optional: compare fitted parameters (NextStat vs ROOT) at fixed-mu points, expressed
            # in NextStat parameter order. This helps diagnose q(mu) differences caused by
            # optimizer basin selection.
            if (
                ns_points
                and root_points
                and len(ns_points) == len(mu_values)
                and len(root_points) == len(mu_values)
                and all(isinstance(p, dict) and ("params" in p) for p in ns_points)
            ):
                # Focus diagnostics on the scan point with max |delta q(mu)|.
                if dq_max_mu is not None:
                    diagnostic["mu_at_max_abs_dq_mu"] = float(dq_max_mu)
                    p_root = _nearest_point(root_points, float(dq_max_mu))
                    p_ns = _nearest_point(ns_points, float(dq_max_mu))
                    if "params" in p_root and "params" in p_ns:
                        root_map = dict(zip(root_names, map(float, p_root["params"])))
                        root_ns = _ns_params_from_root_map(root_map)
                        ns_params = list(map(float, p_ns["params"]))
                        if len(ns_params) == len(root_ns):
                            diffs = [a - b for a, b in zip(ns_params, root_ns)]
                            maxabs = max(abs(d) for d in diffs) if diffs else 0.0
                            l2 = float(sum(d * d for d in diffs) ** 0.5)
                            diagnostic["ns_vs_root_params_dqmax_max_abs"] = float(maxabs)
                            diagnostic["ns_vs_root_params_dqmax_l2"] = float(l2)
                            top = sorted(enumerate(diffs), key=lambda t: abs(t[1]), reverse=True)[:10]
                            diagnostic["ns_vs_root_params_dqmax_top_abs"] = [
                                {
                                    "name": ns_names[j],
                                    "delta": float(d),
                                    "root": float(root_ns[j]),
                                    "nextstat": float(ns_params[j]),
                                }
                                for j, d in top
                            ]

                            # Compare NLL at the two conditional minima (expressed in NextStat space).
                            nll_ns = float(ns_model.nll(ns_params))
                            nll_root_mapped = float(ns_model.nll(root_ns))
                            diagnostic["nll_nextstat_at_nextstat_params_dqmax_mu"] = nll_ns
                            diagnostic["nll_nextstat_at_root_mapped_params_dqmax_mu"] = nll_root_mapped
                            diagnostic["delta_nll_nextstat_minus_root_mapped_dqmax_mu"] = (
                                nll_ns - nll_root_mapped
                            )

                            # Gradient sanity at the mismatch point (POI gradient masked out).
                            try:
                                diagnostic["grad_nextstat_at_dqmax_mu"] = _grad_stats(ns_params)
                                diagnostic["grad_root_mapped_at_dqmax_mu"] = _grad_stats(root_ns)
                            except Exception as e:
                                diagnostic["grad_error"] = f"{e}"
        except Exception as e:
            diagnostic["error"] = f"{e}"

    # Normalize NextStat scan into comparable format.
    #
    # Do not key by float(mu): ROOT and Rust/Python can serialize the same nominal
    # grid point with slightly different binary floats (e.g. 0.3 vs 0.30000000000000004),
    # which would cause spurious KeyErrors.
    diffs = q_diffs
    max_abs_dq = max(abs(d) for _, d in diffs) if diffs else 0.0
    max_abs_dq_mu = dq_max_mu
    top_diffs = sorted(diffs, key=lambda t: abs(t[1]), reverse=True)[:10]
    mu_hat_root = float(root_result["mu_hat"])
    mu_hat_ns = float(ns_scan["mu_hat"])
    d_mu_hat = mu_hat_ns - mu_hat_root

    summary = {
        "input": {
            "mode": "pyhf-json" if args.pyhf_json else "histfactory-xml",
            "measurement": measurement_name,
            "mu_grid": {"start": args.start, "stop": args.stop, "points": args.points},
        },
        "timing_s": {
            "build_workspace": t_build_ws,
            "pyhf_profile_scan": t_pyhf,
            "hist2workspace": t_hist2ws,
            "root_profile_scan_wall": t_root,
            "nextstat_profile_scan": t_ns,
        },
        "timing_summary_s": {
            "reference_root_profile_scan": t_root,
            "reference_pyhf_profile_scan": (t_pyhf or 0.0),
            "reference_total_profile_scan": t_root + (t_pyhf or 0.0),
            "nextstat_profile_scan": t_ns,
            "speedup_vs_root": (t_root / max(t_ns, 1e-12)),
        },
        "artifacts": {
            "root_profile_scan_json": str(root_out),
            "pyhf_profile_scan_json": str(pyhf_out) if pyhf_out else None,
            "nextstat_profile_scan_json": str(ns_out),
        },
        "root": {
            "mu_hat": mu_hat_root,
            "nll_hat": float(root_result["nll_hat"]),
        },
        "pyhf": pyhf_scan,
        "nextstat": {
            "mu_hat": mu_hat_ns,
            "nll_hat": float(ns_scan["nll_hat"]),
        },
        "diagnostic": (diagnostic or None),
        "diff": {
            "mu_hat": d_mu_hat,
            "max_abs_dq_mu": max_abs_dq,
            "mu_at_max_abs_dq_mu": max_abs_dq_mu,
            "top_dq_mu_by_abs": [{"mu": float(mu), "delta_q_mu": float(dq)} for mu, dq in top_diffs],
        },
    }

    out_path = run_dir / "summary.json"
    out_path.write_text(json.dumps(summary, indent=2, sort_keys=True))

    print(json.dumps(summary, indent=2, sort_keys=True))

    if not args.keep:
        # Keep only summary + root scan + allow inspecting intermediate artifacts when needed.
        # (Comment this out if you want to always keep the HistFactory files.)
        pass

    return 0


if __name__ == "__main__":
    raise SystemExit(main())
