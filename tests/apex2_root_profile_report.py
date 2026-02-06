#!/usr/bin/env python3
"""Apex2 ROOT/HistFactory parity runner (profile scan).

This wraps `tests/validate_root_profile_scan.py` and produces a single JSON report:
  - ROOT reference (hist2workspace + RooFit profiling)
  - NextStat `profile_scan`
  - diffs: max_abs_dq_mu, mu_hat delta
  - timings (wall)

It is designed to fail fast with actionable prerequisite messages when ROOT or
`uproot` are missing.

Run:
  PYTHONPATH=bindings/ns-py/python ./.venv/bin/python tests/apex2_root_profile_report.py \\
    --pyhf-json tests/fixtures/simple_workspace.json --measurement GaussExample
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
from typing import Any, Dict, Optional


def _which(name: str) -> Optional[str]:
    return shutil.which(name)


def _latest_run_dir(workdir: Path) -> Optional[Path]:
    runs = sorted(workdir.glob("run_*"), key=lambda p: p.stat().st_mtime, reverse=True)
    return runs[0] if runs else None


def _run(cmd: list[str], *, cwd: Path) -> None:
    p = subprocess.run(
        cmd, cwd=str(cwd), stdout=subprocess.PIPE, stderr=subprocess.STDOUT, text=True
    )
    if p.returncode != 0:
        raise RuntimeError(f"Command failed ({p.returncode}): {' '.join(cmd)}\n{p.stdout}")


def main() -> int:
    ap = argparse.ArgumentParser()
    ap.add_argument("--pyhf-json", type=Path, default=None)
    ap.add_argument("--measurement", type=str, default=None)
    ap.add_argument("--histfactory-xml", type=Path, default=None)
    ap.add_argument("--rootdir", type=Path, default=None)
    ap.add_argument("--start", type=float, default=0.0)
    ap.add_argument("--stop", type=float, default=5.0)
    ap.add_argument("--points", type=int, default=51)
    ap.add_argument("--workdir", type=Path, default=Path("tmp/root_parity"))
    ap.add_argument("--out", type=Path, default=Path("tmp/apex2_root_profile_report.json"))
    ap.add_argument("--dq-atol", type=float, default=1e-3)
    ap.add_argument("--mu-hat-atol", type=float, default=1e-3)
    args = ap.parse_args()

    if bool(args.pyhf_json) == bool(args.histfactory_xml):
        print("Provide exactly one of: --pyhf-json or --histfactory-xml", file=sys.stderr)
        return 2

    if args.pyhf_json and not args.measurement:
        print("--measurement is required with --pyhf-json", file=sys.stderr)
        return 2

    hist2ws = _which("hist2workspace")
    if not hist2ws:
        print(
            "Missing prerequisite: ROOT `hist2workspace` not found in PATH.\n"
            "Install ROOT (HistFactory/RooStats) and make sure your environment is set up.\n"
            "Then rerun this script.",
            file=sys.stderr,
        )
        return 3

    # If we start from pyhf JSON or parse XML -> pyhf, pyhf.writexml/readxml will require uproot.
    # We validate early to give a clean error.
    if args.pyhf_json:
        try:
            import pyhf.writexml  # noqa: F401
        except ModuleNotFoundError:
            print(
                "Missing prerequisite: `uproot` is required for `pyhf.writexml`.\n"
                "Install it and rerun. Example:\n"
                "  pip install uproot\n",
                file=sys.stderr,
            )
            return 3
    else:
        try:
            import pyhf.readxml  # noqa: F401
        except ModuleNotFoundError:
            print(
                "Missing prerequisite: `uproot` is required for `pyhf.readxml`.\n"
                "Install it and rerun. Example:\n"
                "  pip install uproot\n",
                file=sys.stderr,
            )
            return 3

    workdir = args.workdir.resolve()
    workdir.mkdir(parents=True, exist_ok=True)

    validate = Path(__file__).resolve().parent / "validate_root_profile_scan.py"
    cmd = [sys.executable, str(validate)]
    if args.pyhf_json:
        cmd += ["--pyhf-json", str(args.pyhf_json), "--measurement", str(args.measurement)]
    else:
        cmd += ["--histfactory-xml", str(args.histfactory_xml)]
        if args.rootdir:
            cmd += ["--rootdir", str(args.rootdir)]
    cmd += [
        "--start",
        str(args.start),
        "--stop",
        str(args.stop),
        "--points",
        str(args.points),
        "--workdir",
        str(workdir),
    ]

    t0 = time.perf_counter()
    _run(cmd, cwd=Path.cwd())
    t_wall = time.perf_counter() - t0

    run_dir = _latest_run_dir(workdir)
    if not run_dir:
        print("No run directory created under workdir.", file=sys.stderr)
        return 4

    summary_path = run_dir / "summary.json"
    if not summary_path.exists():
        print(f"Missing summary.json under: {run_dir}", file=sys.stderr)
        return 4

    summary = json.loads(summary_path.read_text())
    diff = summary.get("diff", {})
    max_abs_dq_mu = float(diff.get("max_abs_dq_mu", float("nan")))
    d_mu_hat = float(diff.get("mu_hat", float("nan")))

    ok = (abs(max_abs_dq_mu) <= args.dq_atol) and (abs(d_mu_hat) <= args.mu_hat_atol)

    report: Dict[str, Any] = {
        "meta": {
            "timestamp": int(time.time()),
            "wall_s": float(t_wall),
            "workdir": str(workdir),
            "run_dir": str(run_dir),
            "thresholds": {"dq_atol": args.dq_atol, "mu_hat_atol": args.mu_hat_atol},
        },
        "result": {
            "ok": bool(ok),
            "max_abs_dq_mu": float(max_abs_dq_mu),
            "d_mu_hat": float(d_mu_hat),
        },
        "summary": summary,
    }

    args.out.parent.mkdir(parents=True, exist_ok=True)
    args.out.write_text(json.dumps(report, indent=2))

    print(json.dumps(report["result"], indent=2))
    print(f"Wrote: {args.out}")
    return 0 if ok else 2


if __name__ == "__main__":
    raise SystemExit(main())

