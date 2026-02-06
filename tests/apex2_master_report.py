#!/usr/bin/env python3
"""Apex2 master runner: one report for pyhf parity + ROOT parity.

This script combines existing Apex2 runners into a single JSON artifact:
  - pyhf: `tests/apex2_pyhf_validation_report.py`
  - ROOT: `tests/apex2_root_suite_report.py` (runs if prereqs exist, else records skipped)

Run:
  PYTHONPATH=bindings/ns-py/python ./.venv/bin/python tests/apex2_master_report.py
"""

from __future__ import annotations

import argparse
import json
import os
import platform
import subprocess
import sys
import time
from pathlib import Path
from typing import Any, Dict, Optional, Tuple


def _repo_root() -> Path:
    return Path(__file__).resolve().parents[1]


def _with_py_path(env: Dict[str, str]) -> Dict[str, str]:
    # Ensure the editable python package is importable for subprocess calls.
    # Prefer to preserve existing PYTHONPATH order.
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


def _run_json(cmd: list[str], *, cwd: Path, env: Dict[str, str]) -> Tuple[int, str]:
    p = subprocess.run(
        cmd, cwd=str(cwd), env=env, stdout=subprocess.PIPE, stderr=subprocess.STDOUT, text=True
    )
    return p.returncode, p.stdout


def _read_json(path: Path) -> Any:
    return json.loads(path.read_text())


def main() -> int:
    ap = argparse.ArgumentParser()
    ap.add_argument("--out", type=Path, default=Path("tmp/apex2_master_report.json"))
    ap.add_argument("--pyhf-out", type=Path, default=Path("tmp/apex2_pyhf_report.json"))
    ap.add_argument("--root-out", type=Path, default=Path("tmp/apex2_root_suite_report.json"))
    ap.add_argument("--root-cases", type=Path, default=None, help="Cases JSON for ROOT suite.")
    ap.add_argument("--pyhf-sizes", type=str, default="2,16,64,256")
    ap.add_argument("--pyhf-n-random", type=int, default=8)
    ap.add_argument("--pyhf-seed", type=int, default=0)
    ap.add_argument("--pyhf-fit", action="store_true")
    ap.add_argument("--root-prereq-only", action="store_true", help="Only check ROOT prereqs.")
    args = ap.parse_args()

    repo = _repo_root()
    cwd = repo
    env = _with_py_path(os.environ.copy())

    t0 = time.time()
    report: Dict[str, Any] = {
        "meta": {
            "timestamp": int(t0),
            "python": sys.version.split()[0],
            "platform": platform.platform(),
        },
        "pyhf": None,
        "root": None,
    }

    # ------------------------------------------------------------------
    # pyhf runner (always runnable if pyhf installed)
    # ------------------------------------------------------------------
    pyhf_runner = repo / "tests" / "apex2_pyhf_validation_report.py"
    pyhf_cmd = [
        sys.executable,
        str(pyhf_runner),
        "--out",
        str(args.pyhf_out),
        "--sizes",
        args.pyhf_sizes,
        "--n-random",
        str(args.pyhf_n_random),
        "--seed",
        str(args.pyhf_seed),
    ]
    if args.pyhf_fit:
        pyhf_cmd.append("--fit")

    rc_pyhf, out_pyhf = _run_json(pyhf_cmd, cwd=cwd, env=env)
    report["pyhf"] = {
        "returncode": int(rc_pyhf),
        "stdout_tail": out_pyhf[-4000:],
        "report_path": str(args.pyhf_out),
        "report": _read_json(args.pyhf_out) if args.pyhf_out.exists() else None,
    }

    # ------------------------------------------------------------------
    # ROOT suite runner (may be skipped)
    # ------------------------------------------------------------------
    root_runner = repo / "tests" / "apex2_root_suite_report.py"
    root_cmd = [
        sys.executable,
        str(root_runner),
        "--out",
        str(args.root_out),
    ]
    if args.root_cases:
        root_cmd += ["--cases", str(args.root_cases)]
    if args.root_prereq_only:
        root_cmd.append("--prereq-only")

    rc_root, out_root = _run_json(root_cmd, cwd=cwd, env=env)
    root_report = _read_json(args.root_out) if args.root_out.exists() else None
    report["root"] = {
        "returncode": int(rc_root),
        "stdout_tail": out_root[-4000:],
        "report_path": str(args.root_out),
        "report": root_report,
        "prereqs": (root_report or {}).get("meta", {}).get("prereqs") if root_report else None,
    }

    report["meta"]["wall_s"] = float(time.time() - t0)

    args.out.parent.mkdir(parents=True, exist_ok=True)
    args.out.write_text(json.dumps(report, indent=2))

    # Exit code policy: non-zero if pyhf parity fails or ROOT suite fails (if it ran).
    pyhf_ok = (rc_pyhf == 0)
    root_ok_or_skipped = (rc_root == 0) or (root_report and root_report.get("meta", {}).get("prereqs", {}).get("hist2workspace") is False)

    print(f"Wrote: {args.out}")
    if not pyhf_ok:
        return 2
    if not root_ok_or_skipped:
        return 2
    return 0


if __name__ == "__main__":
    raise SystemExit(main())

