#!/usr/bin/env python3
"""Apex2 runner: Cox PH parity vs statsmodels PHReg (optional).

Methodology: Planning -> Exploration -> Execution -> Verification

This runner exists because the core test suite is intentionally dependency-light.
It runs the optional reference parity test:
  - tests/python/test_survival_cox_statsmodels_parity.py

Behavior:
- If `statsmodels` or `numpy` is missing, records status=skipped.
- Otherwise, runs pytest and records status=ok/fail.

Run:
  PYTHONPATH=bindings/ns-py/python ./.venv/bin/python \
    tests/apex2_survival_statsmodels_report.py --out tmp/apex2_survival_statsmodels_report.json
"""

from __future__ import annotations

import argparse
import importlib.util
import os
import platform
import subprocess
import sys
import time
from pathlib import Path
from typing import Any, Dict, Tuple

from _apex2_json import write_report_json


def _repo_root() -> Path:
    return Path(__file__).resolve().parents[1]


def _with_py_path(env: Dict[str, str]) -> Dict[str, str]:
    repo = _repo_root()
    add = str(repo / "bindings" / "ns-py" / "python")
    cur = env.get("PYTHONPATH", "")
    if cur:
        if add in cur.split(os.pathsep):
            return env
        env["PYTHONPATH"] = cur + os.pathsep + add
    else:
        env["PYTHONPATH"] = add
    return env


def _module_available(name: str) -> bool:
    try:
        return importlib.util.find_spec(name) is not None
    except Exception:
        return False


def _run_json(cmd: list[str], *, cwd: Path, env: Dict[str, str]) -> Tuple[int, str]:
    p = subprocess.run(
        cmd, cwd=str(cwd), env=env, stdout=subprocess.PIPE, stderr=subprocess.STDOUT, text=True
    )
    return p.returncode, p.stdout


def main() -> int:
    ap = argparse.ArgumentParser()
    ap.add_argument("--out", type=Path, default=Path("tmp/apex2_survival_statsmodels_report.json"))
    ap.add_argument(
        "--deterministic",
        action="store_true",
        help="Make JSON output deterministic (omit timestamps/timings and stdout tails).",
    )
    args = ap.parse_args()

    repo = _repo_root()
    env = _with_py_path(os.environ.copy())
    t0 = time.time()

    report: Dict[str, Any] = {
        "meta": {
            "timestamp": int(t0),
            "python": sys.version.split()[0],
            "platform": platform.platform(),
            "prereqs": {
                "numpy": _module_available("numpy"),
                "statsmodels": _module_available("statsmodels"),
            },
        },
        "status": None,
        "pytest": None,
    }

    if not report["meta"]["prereqs"]["numpy"] or not report["meta"]["prereqs"]["statsmodels"]:
        report["status"] = "skipped"
        report["reason"] = "missing_dependency:numpy_or_statsmodels"
        write_report_json(args.out, report, deterministic=bool(args.deterministic))
        print(f"Wrote: {args.out}")
        return 0

    cmd = [
        sys.executable,
        "-m",
        "pytest",
        "-q",
        "tests/python/test_survival_cox_statsmodels_parity.py",
    ]
    rc, out = _run_json(cmd, cwd=repo, env=env)
    if bool(args.deterministic):
        out = ""

    report["pytest"] = {
        "returncode": int(rc),
        "stdout_tail": out[-4000:],
        "paths": ["tests/python/test_survival_cox_statsmodels_parity.py"],
    }
    report["status"] = "ok" if rc == 0 else "fail"
    report["meta"]["wall_s"] = float(time.time() - t0)

    write_report_json(args.out, report, deterministic=bool(args.deterministic))
    print(f"Wrote: {args.out}")
    return 0 if rc == 0 else 2


if __name__ == "__main__":
    raise SystemExit(main())

