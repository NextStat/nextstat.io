#!/usr/bin/env python3
"""Root-enabled parity gate for profile scan q(mu).

This is not meant to run in default CI (ROOT is usually unavailable). It is a
small wrapper around `tests/validate_root_profile_scan.py` to produce a strict
pass/fail exit code when prerequisites are present.

Usage (recommended):
  PYTHONPATH=bindings/ns-py/python ./.venv/bin/python tests/root_profile_scan_parity_gate.py
"""

from __future__ import annotations

import json
import shutil
import subprocess
import sys
from pathlib import Path


def _which(name: str) -> bool:
    return bool(shutil.which(name))


def main() -> int:
    if not _which("root") or not _which("hist2workspace"):
        print("SKIP: missing ROOT prereqs (root/hist2workspace).")
        return 0

    validate = Path(__file__).resolve().parent / "validate_root_profile_scan.py"
    outdir = Path("tmp/root_profile_scan_gate").resolve()
    outdir.mkdir(parents=True, exist_ok=True)

    # Keep the gate fast-ish: small grid on a committed realistic fixture.
    cmd = [
        sys.executable,
        str(validate),
        "--pyhf-json",
        "tests/fixtures/tttt-prod_workspace.json",
        "--measurement",
        "FourTop",
        "--start",
        "0.0",
        "--stop",
        "3.0",
        "--points",
        "11",
        "--workdir",
        str(outdir),
        "--keep",
    ]
    p = subprocess.run(cmd, stdout=subprocess.PIPE, stderr=subprocess.STDOUT, text=True)
    if p.returncode != 0:
        print(p.stdout)
        return p.returncode

    # Pick newest run dir.
    runs = sorted(outdir.glob("run_*"), key=lambda p: p.stat().st_mtime, reverse=True)
    if not runs:
        print("ERROR: no run_* directory created")
        return 2
    summary = runs[0] / "summary.json"
    if not summary.exists():
        print("ERROR: missing summary.json")
        return 2

    rep = json.loads(summary.read_text())
    diff = rep.get("diff", {}) or {}
    dq = float(diff.get("max_abs_dq_mu", float("nan")))
    d_mu_hat = float(diff.get("mu_hat", float("nan")))

    dq_atol = 2e-2
    mu_hat_atol = 5e-2
    ok = (abs(dq) <= dq_atol) and (abs(d_mu_hat) <= mu_hat_atol)

    print(f"max_abs_dq_mu={dq} (atol={dq_atol})")
    print(f"mu_hat_delta={d_mu_hat} (atol={mu_hat_atol})")
    return 0 if ok else 1


if __name__ == "__main__":
    raise SystemExit(main())

