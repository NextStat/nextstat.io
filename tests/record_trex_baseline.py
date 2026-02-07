#!/usr/bin/env python3
"""Record a TRExFitter/ROOT reference baseline (external environment).

Two input modes:
1) Start from an existing TREx export directory (contains `combination.xml`).
2) Run TRExFitter from a config, then discover the produced export directory.

This is meant to run in an environment that has:
- ROOT: `root` and `hist2workspace` in PATH
- NextStat Python bindings available via `PYTHONPATH=bindings/ns-py/python`

Outputs (per case):
  tests/baselines/trex/<case>/baseline.json

The baseline uses the same numbers-first schema (`trex_baseline_v0`) as the
NextStat analysis-spec baseline so that the same comparator can be reused.
"""

from __future__ import annotations

import argparse
import hashlib
import json
import os
import platform
import shutil
import socket
import subprocess
import sys
import time
from dataclasses import dataclass
from datetime import datetime, timezone
from pathlib import Path
from typing import Any, Dict, List, Optional, Tuple


def _repo_root() -> Path:
    return Path(__file__).resolve().parents[1]


def _which(name: str) -> Optional[str]:
    return shutil.which(name)


def _run(cmd: List[str], *, cwd: Path, env: Optional[Dict[str, str]] = None) -> Tuple[int, str]:
    p = subprocess.run(
        cmd,
        cwd=str(cwd),
        env=env,
        stdout=subprocess.PIPE,
        stderr=subprocess.STDOUT,
        text=True,
    )
    return p.returncode, p.stdout


def _sha256_file(path: Path) -> str:
    h = hashlib.sha256()
    h.update(path.read_bytes())
    return h.hexdigest()


def _stamp(hostname: str) -> str:
    return f"{hostname}_{datetime.now().strftime('%Y%m%d_%H%M%S')}"


def _cpu_brand() -> str:
    try:
        if platform.system() == "Darwin":
            out = subprocess.check_output(["sysctl", "-n", "machdep.cpu.brand_string"], stderr=subprocess.DEVNULL)
            return out.decode().strip()
        if platform.system() == "Linux":
            with open("/proc/cpuinfo") as f:
                for line in f:
                    if line.startswith("model name"):
                        return line.split(":", 1)[1].strip()
    except Exception:
        pass
    return platform.processor() or "unknown"


def _git_info(repo: Path) -> Dict[str, Any]:
    info: Dict[str, Any] = {}
    try:
        info["commit"] = (
            subprocess.check_output(["git", "rev-parse", "HEAD"], cwd=str(repo), stderr=subprocess.DEVNULL)
            .decode()
            .strip()
        )
        info["commit_short"] = info["commit"][:8]
        info["branch"] = (
            subprocess.check_output(["git", "rev-parse", "--abbrev-ref", "HEAD"], cwd=str(repo), stderr=subprocess.DEVNULL)
            .decode()
            .strip()
        )
        dirty = (
            subprocess.check_output(["git", "status", "--porcelain"], cwd=str(repo), stderr=subprocess.DEVNULL)
            .decode()
            .strip()
        )
        info["dirty"] = bool(dirty)
    except Exception:
        info["error"] = "git_not_available"
    return info


def collect_environment(*, repo: Path, root_version: Optional[str], trex_version: Optional[str]) -> Dict[str, Any]:
    env: Dict[str, Any] = {
        "timestamp": int(time.time()),
        "datetime_utc": datetime.now(timezone.utc).strftime("%Y-%m-%dT%H:%M:%SZ"),
        "hostname": socket.gethostname(),
        "python": sys.version.split()[0],
        "python_full": sys.version.replace("\n", " "),
        "platform": platform.platform(),
        "machine": platform.machine(),
        "system": platform.system(),
        "cpu": _cpu_brand(),
        "git": _git_info(repo),
        "root_version": root_version,
        "trexfitter_version": trex_version,
    }
    return env


def _root_version() -> Optional[str]:
    # Prefer root-config when available.
    if _which("root-config"):
        try:
            out = subprocess.check_output(["root-config", "--version"], stderr=subprocess.DEVNULL).decode().strip()
            return out or None
        except Exception:
            pass
    return None


def _trex_version(trex_cmd: str) -> Optional[str]:
    # Best-effort. Different TRExFitter builds use different flags; try common ones.
    for argv in ([trex_cmd, "--version"], [trex_cmd, "-v"], [trex_cmd, "-V"]):
        try:
            p = subprocess.run(argv, stdout=subprocess.PIPE, stderr=subprocess.STDOUT, text=True)
            if p.returncode == 0 and p.stdout.strip():
                s = p.stdout.strip().splitlines()[0].strip()
                return s[:200]
        except Exception:
            continue
    return None


def _sanitize_case_name(name: str) -> str:
    out = []
    for ch in str(name):
        if ch.isalnum() or ch in ("-", "_", "."):
            out.append(ch)
        else:
            out.append("_")
    s = "".join(out).strip("._")
    return s or "case"


def _discover_latest_combination_xml(search_dir: Path) -> Optional[Path]:
    xs = list(search_dir.rglob("combination.xml"))
    if not xs:
        return None
    xs.sort(key=lambda p: p.stat().st_mtime, reverse=True)
    return xs[0]


@dataclass(frozen=True)
class ExportInput:
    export_dir: Path
    combination_xml: Path


def _resolve_export_input(*, export_dir: Path) -> ExportInput:
    export_dir = export_dir.resolve()
    combo = export_dir / "combination.xml"
    if not combo.exists():
        # Fallback: find any combination.xml under the dir.
        found = _discover_latest_combination_xml(export_dir)
        if found is None:
            raise ValueError(f"No combination.xml found under: {export_dir}")
        combo = found
        export_dir = combo.parent
    return ExportInput(export_dir=export_dir, combination_xml=combo.resolve())


def _require_prereqs(*, need_trex: bool, trex_cmd: str) -> Tuple[bool, Dict[str, Any]]:
    prereq = {
        "root": bool(_which("root")),
        "hist2workspace": bool(_which("hist2workspace")),
        "root-config": bool(_which("root-config")),
        "trexfitter": bool(_which(trex_cmd)) if need_trex else None,
    }
    ok = bool(prereq["root"]) and bool(prereq["hist2workspace"]) and (prereq["trexfitter"] is not False)
    return ok, prereq


def _hist2workspace(*, combination_xml: Path) -> Path:
    hist2ws = _which("hist2workspace")
    if not hist2ws:
        raise RuntimeError("Missing ROOT prerequisite: hist2workspace not found in PATH")

    workdir = combination_xml.parent
    before = {p.resolve() for p in workdir.rglob("*.root")}
    rc, out = _run([hist2ws, combination_xml.name], cwd=workdir)
    if rc != 0:
        raise RuntimeError(f"hist2workspace failed (exit {rc}). Output:\n{out}")

    after = {p.resolve() for p in workdir.rglob("*.root")}
    new_root_files = sorted((after - before), key=lambda p: p.stat().st_mtime, reverse=True)
    new_root_files = [p for p in new_root_files if p.name != "data.root"]
    if not new_root_files:
        raise RuntimeError("hist2workspace did not create a new workspace ROOT file (only data.root found)")
    return new_root_files[0]


def _write_root_macro_fit(macro_path: Path, *, root_workspace_file: Path, out_json: Path) -> None:
    def _c_str(s: str) -> str:
        return s.replace("\\", "\\\\").replace('"', '\\"')

    root_path = _c_str(root_workspace_file.as_posix())
    out_path = _c_str(out_json.as_posix())

    macro = r"""
#include <TFile.h>
#include <TKey.h>
#include <TSystem.h>

#include <memory>
#include <stdexcept>

#include <RooWorkspace.h>
#include <RooAbsPdf.h>
#include <RooAbsData.h>
#include <RooAbsReal.h>
#include <RooArgSet.h>
#include <RooRealVar.h>
#include <RooMinimizer.h>
#include <RooFitResult.h>

#include <RooStats/ModelConfig.h>

#include <algorithm>
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
  if (auto* o = w.obj("ModelConfig")) return dynamic_cast<RooStats::ModelConfig*>(o);
  if (auto* o = w.obj("modelConfig")) return dynamic_cast<RooStats::ModelConfig*>(o);
  return nullptr;
}}

static RooAbsData* find_data(RooWorkspace& w) {{
  if (auto* d = w.data("obsData")) return d;
  if (auto* d = w.data("data")) return d;
  auto all = w.allData();
  if (all.getSize() == 0) return nullptr;
  auto* obj = all.first();
  if (!obj) return nullptr;
  return w.data(obj->GetName());
}}

static int minimize_nll(RooAbsReal& nll) {{
  RooMinimizer m(nll);
  m.setPrintLevel(-1);
  m.setStrategy(0);
  m.optimizeConst(2);
  int status = m.minimize("Minuit2", "Migrad");
  return status;
}}

static std::string escape_json(const std::string& s) {{
  std::string out;
  out.reserve(s.size()+8);
  for (char c : s) {{
    if (c == '\\\\') out += "\\\\\\\\";
    else if (c == '\"') out += "\\\\\"";
    else if (c == '\\n') out += "\\\\n";
    else out += c;
  }}
  return out;
}}

void fit() {{
  gSystem->Load("libRooFit");
  gSystem->Load("libRooStats");

  const char* root_path = "__ROOT_PATH__";
  const char* out_path = "__OUT_PATH__";

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
    throw std::runtime_error("ModelConfig not found (expected 'ModelConfig').");
  }}
  RooAbsPdf* pdf = mc->GetPdf();
  if (!pdf) {{
    throw std::runtime_error("ModelConfig has no PDF.");
  }}
  RooAbsData* data = find_data(*w);
  if (!data) {{
    throw std::runtime_error("Data not found in workspace.");
  }}

  std::unique_ptr<RooAbsReal> nll(pdf->createNLL(*data));
  int status = minimize_nll(*nll);
  double nll_hat = nll->getVal();

  RooFitResult* fr = nullptr;
  try {{
    RooMinimizer m(*nll);
    m.setPrintLevel(-1);
    m.setStrategy(0);
    m.optimizeConst(2);
    m.minimize("Minuit2", "Migrad");
    fr = m.save();
  }} catch (...) {{
    fr = nullptr;
  }}

  struct P {{ std::string name; double val; double err; }};
  std::vector<P> ps;
  if (fr) {{
    auto vars = fr->floatParsFinal();
    for (int i = 0; i < vars.getSize(); i++) {{
      auto* v = dynamic_cast<RooRealVar*>(vars.at(i));
      if (!v) continue;
      ps.push_back(P{{v->GetName(), v->getVal(), v->getError()}});
    }}
  }}
  std::sort(ps.begin(), ps.end(), [](const P& a, const P& b) {{ return a.name < b.name; }});

  std::ofstream out(out_path);
  out << std::setprecision(16);
  out << "{\\n";
  out << "  \\\"status\\\": " << status << ",\\n";
  out << "  \\\"nll_hat\\\": " << nll_hat << ",\\n";
  out << "  \\\"twice_nll\\\": " << (2.0*nll_hat) << ",\\n";
  out << "  \\\"parameters\\\": [\\n";
  for (size_t i = 0; i < ps.size(); i++) {{
    out << "    {\\\"name\\\": \\\"" << escape_json(ps[i].name) << "\\\", \\\"value\\\": " << ps[i].val << ", \\\"uncertainty\\\": " << ps[i].err << "}";
    if (i+1 < ps.size()) out << ",";
    out << "\\n";
  }}
  out << "  ]\\n";
  out << "}\\n";
  out.close();
}}
"""
    macro = macro.replace("__ROOT_PATH__", root_path).replace("__OUT_PATH__", out_path)
    macro_path.write_text(macro)


def _run_root_macro(macro_path: Path, *, cwd: Path) -> None:
    root = _which("root")
    if not root:
        raise RuntimeError("Missing ROOT prerequisite: root not found in PATH")
    rc, out = _run([root, "-b", "-q", macro_path.name], cwd=cwd)
    if rc != 0:
        raise RuntimeError(f"ROOT macro failed (exit {rc}). Output:\n{out}")


def _load_json(path: Path) -> Any:
    return json.loads(path.read_text())


def _write_json(path: Path, obj: Any) -> None:
    path.parent.mkdir(parents=True, exist_ok=True)
    path.write_text(json.dumps(obj, indent=2))


def _build_expected_data(*, workspace_json_text: str, root_fit: Dict[str, Any]) -> Dict[str, Any]:
    # Use NextStat bindings to compute expected_data in the pyhf ordering.
    try:
        import nextstat._core as _core  # type: ignore
    except Exception as e:
        raise RuntimeError(f"Missing NextStat Python bindings (set PYTHONPATH=bindings/ns-py/python): {e}")

    model = _core.HistFactoryModel.from_workspace(workspace_json_text)
    names = list(model.parameter_names())
    init = [float(x) for x in model.suggested_init()]
    name_to_index = {str(n): i for i, n in enumerate(names)}

    vec = list(init)
    params = root_fit.get("parameters") or []
    if not isinstance(params, list):
        raise RuntimeError("root fit JSON missing `parameters` list")

    missing: List[str] = []
    for p in params:
        if not isinstance(p, dict):
            continue
        n = str(p.get("name") or "")
        if not n:
            continue
        if n not in name_to_index:
            missing.append(n)
            continue
        vec[name_to_index[n]] = float(p.get("value"))

    # Note: missing params are left at suggested_init.
    exp_main = [float(x) for x in model.expected_data(vec, include_auxdata=False)]
    exp_with_aux = [float(x) for x in model.expected_data(vec, include_auxdata=True)]
    return {
        "pyhf_main": exp_main,
        "pyhf_with_aux": exp_with_aux,
        "missing_params": sorted(set(missing)),
    }


def _import_histfactory_to_workspace_json(*, combination_xml: Path, rootdir: Path) -> str:
    # Use pyhf.readxml for conversion (external env can install `uproot`).
    try:
        import pyhf.readxml as readxml  # type: ignore
    except Exception as e:
        raise RuntimeError(
            f"Missing dependency for HistFactory XML import: {e}. "
            "Install extras: `pip install -e \"bindings/ns-py[validation]\"`"
        )
    ws = readxml.parse(str(combination_xml), rootdir=str(rootdir), track_progress=False)
    return json.dumps(ws)


def record_one_case(
    *,
    case_name: str,
    export: Optional[ExportInput],
    out_root: Path,
    rootdir: Optional[Path],
    trex_config: Optional[Path],
    trex_cmdline: Optional[List[str]],
    seed: Optional[int],
    keep_work: bool,
) -> Path:
    repo = _repo_root()

    out_dir = out_root / _sanitize_case_name(case_name)
    out_dir.mkdir(parents=True, exist_ok=True)
    baseline_path = out_dir / "baseline.json"

    root_version = _root_version()
    trex_version = None
    if trex_cmdline:
        trex_version = _trex_version(trex_cmdline[0])

    environment = collect_environment(repo=repo, root_version=root_version, trex_version=trex_version)

    # Optionally run TRExFitter (best-effort; baseline can still be recorded from existing exports).
    trex_log = None
    if trex_cmdline and trex_config is not None:
        workdir = out_dir / "trex_work"
        workdir.mkdir(parents=True, exist_ok=True)
        cmd = list(trex_cmdline) + [str(trex_config)]
        rc, out = _run(cmd, cwd=workdir, env=os.environ.copy())
        trex_log = out[-20000:] if out else ""
        if rc != 0:
            raise RuntimeError(f"TRExFitter run failed (exit {rc}). Output:\n{trex_log}")
        if export is None:
            found = _discover_latest_combination_xml(workdir)
            if found is None:
                raise RuntimeError(f"TRExFitter run completed, but no combination.xml found under: {workdir}")
            export = _resolve_export_input(export_dir=found.parent)

    if export is None:
        raise RuntimeError("Internal error: export is None after resolution")

    # Record input hashes (combination.xml + root files in the export dir).
    input_hashes: List[Dict[str, Any]] = []
    input_hashes.append(
        {
            "path": str(export.combination_xml),
            "rel": "combination.xml",
            "bytes": export.combination_xml.stat().st_size,
            "sha256": _sha256_file(export.combination_xml),
        }
    )
    for p in sorted(export.export_dir.glob("*.root"), key=lambda x: x.name):
        try:
            input_hashes.append(
                {
                    "path": str(p),
                    "rel": str(p.relative_to(export.export_dir)),
                    "bytes": p.stat().st_size,
                    "sha256": _sha256_file(p),
                }
            )
        except Exception:
            continue

    # ROOT: hist2workspace + free fit via macro.
    # NOTE: This runs in-place in the export dir (hist2workspace writes ROOT files there).
    root_ws_file = _hist2workspace(combination_xml=export.combination_xml)
    fit_out = out_dir / "root_fit.json"
    macro_path = out_dir / "root_fit.C"
    _write_root_macro_fit(macro_path, root_workspace_file=root_ws_file, out_json=fit_out)
    _run_root_macro(macro_path, cwd=out_dir)
    root_fit = _load_json(fit_out)

    # Expected data: convert HistFactory XML -> pyhf JSON workspace; evaluate with NextStat.
    resolved_rootdir = (rootdir or export.export_dir).resolve()
    ws_text = _import_histfactory_to_workspace_json(combination_xml=export.combination_xml, rootdir=resolved_rootdir)
    exp = _build_expected_data(workspace_json_text=ws_text, root_fit=root_fit)

    # Build baseline payload (numbers-first).
    # Align meta with `docs/schemas/trex/baseline_v0.schema.json`.
    tool_versions: Dict[str, Any] = {"root": root_version, "trexfitter": trex_version}
    try:
        import nextstat  # type: ignore

        tool_versions["nextstat"] = getattr(nextstat, "__version__", None)
    except Exception:
        tool_versions["nextstat"] = None

    baseline: Dict[str, Any] = {
        "schema_version": "trex_baseline_v0",
        "meta": {
            "created_at": datetime.now(timezone.utc).strftime("%Y-%m-%dT%H:%M:%SZ"),
            # Determinism in this context means we ran a single reference fit and store stable ordering.
            "deterministic": True,
            "threads": 1,
            "seed": int(seed) if seed is not None else None,
            "tool_versions": tool_versions,
            "inputs": {
                "export_dir": str(export.export_dir),
                "combination_xml": str(export.combination_xml),
                "rootdir": str(resolved_rootdir),
                "hashes": input_hashes,
            },
            # Extra context (non-contract) for reproducibility.
            "baseline_env": environment,
            "reference": {"mode": "trexfitter" if trex_cmdline else "histfactory_export"},
        },
        "fit": {
            "twice_nll": float(root_fit.get("twice_nll")),
            "parameters": list(root_fit.get("parameters") or []),
            "covariance": None,
        },
        "expected_data": {"pyhf_main": exp["pyhf_main"], "pyhf_with_aux": exp["pyhf_with_aux"]},
    }

    # Extra debug info (non-contract).
    baseline["meta"]["root_fit"] = {
        "status": root_fit.get("status"),
        "nll_hat": root_fit.get("nll_hat"),
        "missing_params": exp.get("missing_params"),
    }
    if trex_log is not None:
        baseline["meta"]["trexfitter_log_tail"] = trex_log

    _write_json(baseline_path, baseline)
    return baseline_path


def main() -> int:
    ap = argparse.ArgumentParser()
    ap.add_argument("--out-root", type=Path, default=Path("tests/baselines/trex"))
    ap.add_argument("--case", type=str, default=None, help="Case name (default: derived from export dir).")
    ap.add_argument("--export-dir", type=Path, default=None, help="TREx export directory containing combination.xml.")
    ap.add_argument("--rootdir", type=Path, default=None, help="ROOT directory for pyhf.readxml (default: export dir).")
    ap.add_argument("--trex-config", type=Path, default=None, help="TRExFitter config path (optional).")
    ap.add_argument(
        "--trex-cmd",
        type=str,
        default="trex-fitter",
        help="TRExFitter executable name/path (used only with --trex-config).",
    )
    ap.add_argument(
        "--trex-args",
        type=str,
        default="",
        help="Extra TRExFitter args (space-separated). Recorder appends the config path as last arg.",
    )
    ap.add_argument("--seed", type=int, default=None)
    ap.add_argument("--keep-work", action="store_true")
    ap.add_argument("--prereq-only", action="store_true")
    args = ap.parse_args()

    if args.export_dir is None and args.trex_config is None:
        print("ERROR: provide --export-dir or --trex-config", file=sys.stderr)
        return 2

    need_trex = args.trex_config is not None
    trex_cmdline = None
    if need_trex:
        trex_cmdline = [str(args.trex_cmd)] + ([a for a in str(args.trex_args).split() if a] if args.trex_args else [])

    ok, prereq = _require_prereqs(need_trex=need_trex, trex_cmd=str(args.trex_cmd))
    if args.prereq_only:
        print(json.dumps({"ok": ok, "prereqs": prereq}, indent=2, sort_keys=True))
        return 0 if ok else 3
    if not ok:
        missing = [k for k, v in prereq.items() if v is False]
        print(f"ERROR: missing prereqs: {missing}. This recorder must run in an external env with ROOT/TREx.", file=sys.stderr)
        return 3

    export = _resolve_export_input(export_dir=Path(args.export_dir)) if args.export_dir is not None else None
    case_name = args.case or (export.export_dir.name if export is not None else "trex_case")

    try:
        baseline_path = record_one_case(
            case_name=str(case_name),
            export=export,
            out_root=Path(args.out_root),
            rootdir=args.rootdir,
            trex_config=args.trex_config,
            trex_cmdline=trex_cmdline,
            seed=args.seed,
            keep_work=bool(args.keep_work),
        )
    except Exception as e:
        print(f"ERROR: baseline recording failed: {e}", file=sys.stderr)
        return 4

    print(f"Wrote: {baseline_path}")
    return 0


if __name__ == "__main__":
    raise SystemExit(main())
