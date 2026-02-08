#!/usr/bin/env python3
"""Apex2 runner: ROOT/HistFactory parity on multiple cases.

This is meant to be executed on a machine with ROOT (hist2workspace + RooFit/RooStats)
available. It reuses `tests/validate_root_profile_scan.py` as the underlying engine,
and aggregates results into a single JSON report.

It can also run in "prereq-only" mode to quickly validate the environment.

Cases can be provided either:
  1) via `--cases <json>` file (recommended), or
  2) using built-in defaults (simple fixture) if `--cases` is omitted.

Example cases file:
{
  "cases": [
    {
      "name": "simple_fixture",
      "mode": "pyhf-json",
      "pyhf_json": "tests/fixtures/simple_workspace.json",
      "measurement": "GaussExample",
      "mu_grid": {"start": 0.0, "stop": 5.0, "points": 51}
    },
    {
      "name": "trexfitter_export",
      "mode": "histfactory-xml",
      "histfactory_xml": "/abs/path/to/combination.xml",
      "rootdir": "/abs/path/to",
      "mu_grid": {"start": 0.0, "stop": 5.0, "points": 51}
    }
  ]
}
"""

from __future__ import annotations

import argparse
import json
import shutil
import subprocess
import sys
import time
from pathlib import Path
from typing import Any, Dict, List, Optional, Tuple

from _apex2_json import write_report_json


def _which(name: str) -> Optional[str]:
    return shutil.which(name)


def _sanitize_name(name: str) -> str:
    out = []
    for ch in name:
        if ch.isalnum() or ch in ("-", "_", "."):
            out.append(ch)
        else:
            out.append("_")
    s = "".join(out).strip("._")
    return s or "case"


def _latest_run_dir(workdir: Path) -> Optional[Path]:
    runs = sorted(workdir.glob("run_*"), key=lambda p: p.stat().st_mtime, reverse=True)
    return runs[0] if runs else None


def _run(cmd: List[str], *, cwd: Path) -> Tuple[int, str]:
    p = subprocess.run(
        cmd, cwd=str(cwd), stdout=subprocess.PIPE, stderr=subprocess.STDOUT, text=True
    )
    return p.returncode, p.stdout


def _load_cases(path: Optional[Path]) -> List[Dict[str, Any]]:
    if path is None:
        return [
            {
                "name": "simple_fixture",
                "mode": "pyhf-json",
                "pyhf_json": "tests/fixtures/simple_workspace.json",
                "measurement": "GaussExample",
                "mu_grid": {"start": 0.0, "stop": 5.0, "points": 51},
            }
        ]

    data = json.loads(path.read_text())
    cases = data.get("cases")
    if not isinstance(cases, list) or not cases:
        raise SystemExit("cases file must contain non-empty list under key 'cases'")
    return cases


def _check_prereqs(*, need_uproot: bool) -> Dict[str, Any]:
    prereq: Dict[str, Any] = {
        "hist2workspace": bool(_which("hist2workspace")),
        "root": bool(_which("root")),
        "uproot": None,
    }

    if need_uproot:
        try:
            import uproot  # noqa: F401
        except ModuleNotFoundError:
            prereq["uproot"] = False
        else:
            prereq["uproot"] = True

    return prereq


def main() -> int:
    ap = argparse.ArgumentParser()
    ap.add_argument("--cases", type=Path, default=None, help="Path to JSON cases file.")
    ap.add_argument(
        "--case-index",
        type=int,
        default=None,
        help="Run only a single case selected by 0-based index from --cases.",
    )
    ap.add_argument(
        "--case-name",
        type=str,
        default=None,
        help="Run only a single case with matching 'name' from --cases.",
    )
    ap.add_argument("--workdir", type=Path, default=Path("tmp/root_parity_suite"))
    ap.add_argument("--out", type=Path, default=Path("tmp/apex2_root_suite_report.json"))
    ap.add_argument("--keep-going", action="store_true", help="Keep running after failures.")
    ap.add_argument("--prereq-only", action="store_true", help="Only validate prerequisites, do not run.")
    ap.add_argument("--dq-atol", type=float, default=1e-3)
    # ROOT (Minuit2) and NextStat (L-BFGS-B) can disagree slightly on mu_hat at ~1e-3 scale.
    # Keep this loose enough to avoid false negatives across ROOT versions.
    ap.add_argument("--mu-hat-atol", type=float, default=2e-3)
    ap.add_argument(
        "--deterministic",
        action="store_true",
        help="Make JSON output deterministic (stable ordering; omit timestamps/timings).",
    )
    args = ap.parse_args()

    if args.case_index is not None and args.case_name is not None:
        raise SystemExit("--case-index and --case-name are mutually exclusive")

    cases = _load_cases(args.cases)
    if args.case_index is not None:
        idx = int(args.case_index)
        if idx < 0 or idx >= len(cases):
            raise SystemExit(f"--case-index out of range: {idx} (n_cases={len(cases)})")
        cases = [cases[idx]]
    if args.case_name is not None:
        name = str(args.case_name)
        matches = [c for c in cases if str(c.get("name") or "") == name]
        if not matches:
            raise SystemExit(f"--case-name not found: {name}")
        cases = [matches[0]]

    workdir = args.workdir.resolve()
    workdir.mkdir(parents=True, exist_ok=True)

    # Only pyhf-json mode needs `uproot` (via pyhf.writexml). HistFactory XML mode runs without it.
    need_uproot = any(c.get("mode") == "pyhf-json" for c in cases)
    prereq = _check_prereqs(need_uproot=need_uproot)

    report: Dict[str, Any] = {
        "meta": {
            "timestamp": int(time.time()),
            "cases_file": str(args.cases) if args.cases else None,
            "case_index": int(args.case_index) if args.case_index is not None else None,
            "case_name": str(args.case_name) if args.case_name is not None else None,
            "workdir": str(workdir),
            "thresholds": {"dq_atol": args.dq_atol, "mu_hat_atol": args.mu_hat_atol},
            "prereqs": prereq,
        },
        "cases": [],
        "summary": {},
    }

    if args.prereq_only:
        write_report_json(args.out, report, deterministic=bool(args.deterministic))
        print(json.dumps(report["meta"]["prereqs"], indent=2, sort_keys=True))
        print(f"Wrote: {args.out}")
        # Exit code: 0 only if all required prereqs are satisfied.
        ok = bool(prereq["hist2workspace"]) and bool(prereq["root"]) and (prereq["uproot"] is not False)
        return 0 if ok else 3

    validate = Path(__file__).resolve().parent / "validate_root_profile_scan.py"

    any_failed = False
    for case in cases:
        name = str(case.get("name") or "case")
        mode = case.get("mode")
        if mode not in ("pyhf-json", "histfactory-xml"):
            any_failed = True
            report["cases"].append(
                {"name": name, "status": "error", "reason": f"unknown_mode:{mode}"}
            )
            if not args.keep_going:
                break
            continue

        # Per-case prereq checks
        if not prereq["hist2workspace"]:
            any_failed = True
            report["cases"].append(
                {
                    "name": name,
                    "status": "skipped",
                    "reason": "missing_hist2workspace",
                }
            )
            if not args.keep_going:
                break
            continue

        if not prereq["root"]:
            any_failed = True
            report["cases"].append(
                {
                    "name": name,
                    "status": "skipped",
                    "reason": "missing_root",
                }
            )
            if not args.keep_going:
                break
            continue

        if mode == "pyhf-json" and prereq["uproot"] is False:
            any_failed = True
            report["cases"].append(
                {"name": name, "status": "skipped", "reason": "missing_uproot"}
            )
            if not args.keep_going:
                break
            continue

        case_dir = workdir / _sanitize_name(name)
        case_dir.mkdir(parents=True, exist_ok=True)

        mu_grid = case.get("mu_grid") or {}
        start = float(mu_grid.get("start", 0.0))
        stop = float(mu_grid.get("stop", 5.0))
        points = int(mu_grid.get("points", 51))

        cmd = [sys.executable, str(validate)]
        if mode == "pyhf-json":
            pyhf_json = case.get("pyhf_json")
            measurement = case.get("measurement")
            if not pyhf_json or not measurement:
                any_failed = True
                report["cases"].append(
                    {
                        "name": name,
                        "status": "error",
                        "reason": "missing_pyhf_json_or_measurement",
                    }
                )
                if not args.keep_going:
                    break
                continue
            cmd += ["--pyhf-json", str(pyhf_json), "--measurement", str(measurement)]
        else:
            histfactory_xml = case.get("histfactory_xml")
            if not histfactory_xml:
                any_failed = True
                report["cases"].append(
                    {"name": name, "status": "error", "reason": "missing_histfactory_xml"}
                )
                if not args.keep_going:
                    break
                continue
            cmd += ["--histfactory-xml", str(histfactory_xml)]
            rootdir = case.get("rootdir")
            if rootdir:
                cmd += ["--rootdir", str(rootdir)]

        cmd += [
            "--start",
            str(start),
            "--stop",
            str(stop),
            "--points",
            str(points),
            "--workdir",
            str(case_dir),
        ]

        t0 = time.perf_counter()
        rc, out = _run(cmd, cwd=Path.cwd())
        wall_s = time.perf_counter() - t0

        run_dir = _latest_run_dir(case_dir)
        summary_path = (run_dir / "summary.json") if run_dir else None

        if rc != 0 or not summary_path or not summary_path.exists():
            any_failed = True
            report["cases"].append(
                {
                    "name": name,
                    "status": "error",
                    "reason": "validate_root_profile_scan_failed",
                    "returncode": int(rc),
                    "wall_s": float(wall_s),
                    "run_dir": str(run_dir) if run_dir else None,
                    "stdout_tail": out[-4000:],
                }
            )
            if not args.keep_going:
                break
            continue

        summary = json.loads(summary_path.read_text())
        diff = summary.get("diff", {})
        max_abs_dq_mu = float(diff.get("max_abs_dq_mu", float("nan")))
        d_mu_hat = float(diff.get("mu_hat", float("nan")))
        ok = (abs(max_abs_dq_mu) <= args.dq_atol) and (abs(d_mu_hat) <= args.mu_hat_atol)
        if not ok:
            any_failed = True

        timing_s = summary.get("timing_s", {}) or {}
        root_scan_wall = timing_s.get("root_profile_scan_wall")
        ns_scan = timing_s.get("nextstat_profile_scan")
        speedup_ns_vs_root_scan = None
        try:
            if root_scan_wall is not None and ns_scan is not None:
                root_scan_wall_f = float(root_scan_wall)
                ns_scan_f = float(ns_scan)
                if ns_scan_f > 0.0:
                    speedup_ns_vs_root_scan = root_scan_wall_f / ns_scan_f
        except Exception:
            speedup_ns_vs_root_scan = None

        report["cases"].append(
            {
                "name": name,
                "status": "ok" if ok else "fail",
                "wall_s": float(wall_s),
                "run_dir": str(run_dir),
                "summary_path": str(summary_path),
                "diff": {"max_abs_dq_mu": max_abs_dq_mu, "d_mu_hat": d_mu_hat},
                "timing_s": timing_s,
                "perf": {
                    "speedup_nextstat_vs_root_scan": speedup_ns_vs_root_scan,
                },
            }
        )

    n_cases = len(report["cases"])
    n_ok = sum(1 for c in report["cases"] if c.get("status") == "ok")
    n_fail = sum(1 for c in report["cases"] if c.get("status") == "fail")
    n_skip = sum(1 for c in report["cases"] if c.get("status") == "skipped")
    n_err = sum(1 for c in report["cases"] if c.get("status") == "error")

    report["summary"] = {
        "n_cases": int(n_cases),
        "n_ok": int(n_ok),
        "n_fail": int(n_fail),
        "n_skipped": int(n_skip),
        "n_error": int(n_err),
    }

    write_report_json(args.out, report, deterministic=bool(args.deterministic))

    print(
        json.dumps(
            {
                "n_cases": n_cases,
                "ok": n_ok,
                "fail": n_fail,
                "skipped": n_skip,
                "error": n_err,
            },
            indent=2,
        )
    )
    print(f"Wrote: {args.out}")

    return 0 if not any_failed else 2


if __name__ == "__main__":
    raise SystemExit(main())
