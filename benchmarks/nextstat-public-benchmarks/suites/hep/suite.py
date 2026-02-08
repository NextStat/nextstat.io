#!/usr/bin/env python3
"""HEP suite runner (seed).

Runs multiple NLL parity+timing cases and writes:
- per-case JSON files (benchmark_result_v1)
- a suite index JSON (benchmark_suite_result_v1)
"""

from __future__ import annotations

import argparse
import hashlib
import json
import os
from pathlib import Path

import pyhf

import nextstat


def sha256_file(path: Path) -> str:
    h = hashlib.sha256()
    with path.open("rb") as f:
        for chunk in iter(lambda: f.read(1024 * 1024), b""):
            h.update(chunk)
    return h.hexdigest()


def load_workspace(path: Path) -> dict:
    return json.loads(path.read_text())


def make_synthetic_shapesys_workspace(n_bins: int) -> dict:
    signal = {
        "name": "signal",
        "data": [5.0] * n_bins,
        "modifiers": [{"name": "mu", "type": "normfactor", "data": None}],
    }
    bkg = {
        "name": "background",
        "data": [50.0] * n_bins,
        "modifiers": [{"name": "uncorr_bkguncrt", "type": "shapesys", "data": [5.0] * n_bins}],
    }
    return {
        "channels": [{"name": "c", "samples": [signal, bkg]}],
        "observations": [{"name": "c", "data": [53.0] * n_bins}],
        "measurements": [{"name": "m", "config": {"poi": "mu", "parameters": []}}],
        "version": "1.0.0",
    }


def run_case_with_imported_runpy(case_id: str, workspace: dict, measurement_name: str, *, deterministic: bool, out_path: Path) -> int:
    # Import the single-case runner as a module so we reuse its benchmark logic and JSON format.
    from . import run as run_one  # type: ignore

    # Emulate CLI args via direct call.
    argv = [
        "--out",
        str(out_path),
        "--measurement-name",
        measurement_name,
    ]
    if deterministic:
        argv.append("--deterministic")

    # The single-case runner expects a workspace file path; write a temp copy next to the output.
    tmp_ws = out_path.with_suffix(".workspace.json")
    tmp_ws.write_text(json.dumps(workspace, indent=2, sort_keys=True) + "\n")
    argv.extend(["--workspace", str(tmp_ws)])

    try:
        rc = run_one.main_from_argv(argv, case_override=case_id)
    finally:
        try:
            tmp_ws.unlink()
        except OSError:
            pass
    return rc


def main() -> int:
    ap = argparse.ArgumentParser()
    ap.add_argument("--out-dir", required=True, help="Output directory.")
    ap.add_argument("--deterministic", action="store_true")
    ap.add_argument(
        "--sizes",
        default="2,16,64,256",
        help="Comma-separated synthetic bin counts (shapesys) to include.",
    )
    ap.add_argument(
        "--cases",
        default="simple,complex,synthetic",
        help="Comma-separated case groups: simple,complex,synthetic.",
    )
    args = ap.parse_args()

    out_dir = Path(args.out_dir).resolve()
    cases_dir = out_dir / "cases"
    cases_dir.mkdir(parents=True, exist_ok=True)

    deterministic = bool(args.deterministic)

    case_groups = [x.strip() for x in args.cases.split(",") if x.strip()]
    sizes = [int(x.strip()) for x in args.sizes.split(",") if x.strip()]

    suite_cases: list[tuple[str, dict, str]] = []
    ds_dir = Path(__file__).resolve().parent / "datasets"
    if "simple" in case_groups:
        suite_cases.append(("simple_workspace_nll", load_workspace(ds_dir / "simple_workspace.json"), "GaussExample"))
    if "complex" in case_groups:
        complex_path = ds_dir / "complex_workspace.json"
        if complex_path.exists():
            suite_cases.append(("complex_workspace_nll", load_workspace(complex_path), "measurement"))
    if "synthetic" in case_groups:
        for n in sizes:
            suite_cases.append((f"synthetic_shapesys_{n}", make_synthetic_shapesys_workspace(n), "m"))

    index_cases = []
    n_ok = 0
    worst_abs = 0.0
    worst_case = "none"

    for case_id, ws, measurement in suite_cases:
        out_path = cases_dir / f"{case_id}.json"
        rc = run_case_with_imported_runpy(case_id, ws, measurement, deterministic=deterministic, out_path=out_path)
        if rc != 0:
            # Still include it in the index if it exists, but fail overall.
            pass
        obj = json.loads(out_path.read_text())
        sha = sha256_file(out_path)
        parity_ok = bool(obj.get("parity", {}).get("ok", False))
        abs_diff = float(obj.get("parity", {}).get("abs_diff", 0.0))
        speedup = float(obj.get("timing", {}).get("speedup_pyhf_over_nextstat", 0.0))
        if parity_ok:
            n_ok += 1
        if abs_diff >= worst_abs:
            worst_abs = abs_diff
            worst_case = case_id
        index_cases.append(
            {
                "case": case_id,
                "path": os.path.relpath(out_path, out_dir),
                "sha256": sha,
                "parity_ok": parity_ok,
                "abs_diff": abs_diff,
                "speedup_pyhf_over_nextstat": speedup,
            }
        )

    index = {
        "schema_version": "nextstat.benchmark_suite_result.v1",
        "suite": "hep",
        "deterministic": deterministic,
        "meta": {
            "python": obj["meta"]["python"] if index_cases else "unknown",
            "platform": obj["meta"]["platform"] if index_cases else "unknown",
            "pyhf_version": pyhf.__version__,
            "nextstat_version": nextstat.__version__,
        },
        "cases": index_cases,
        "summary": {
            "n_cases": len(index_cases),
            "n_ok": n_ok,
            "worst_abs_diff": worst_abs,
            "worst_case": worst_case,
        },
    }

    index_path = out_dir / "hep_suite.json"
    index_path.write_text(json.dumps(index, indent=2, sort_keys=True) + "\n")

    return 0 if n_ok == len(index_cases) else 2


if __name__ == "__main__":
    raise SystemExit(main())

