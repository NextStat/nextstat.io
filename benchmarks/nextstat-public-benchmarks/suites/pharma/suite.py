#!/usr/bin/env python3
"""Pharma suite runner (seed).

Runs multiple generated PK/NLME cases and writes:
- per-case JSON files (pharma_benchmark_result_v2)
- a suite index JSON (pharma_benchmark_suite_result_v2)

Cases:
- pk_1c_oral_nobs_{8,32,128}:         1-cpt oral NLL timing
- pk_2cpt_iv_nobs_{32,64}:            2-cpt IV NLL timing + fit
- pk_2cpt_oral_nobs_{32,64}:          2-cpt oral NLL timing + fit
- nlme_1c_oral_nsub_{4,16}:           NLME LogDensityModel NLL timing
- foce_1c_oral_nsub_{30,100}:         FOCE population fit + recovery
- saem_1c_oral_nsub_{30,100}:         SAEM population fit + recovery
"""

from __future__ import annotations

import argparse
import hashlib
import json
import os
import platform
import subprocess
import sys
from pathlib import Path
from typing import Any

import nextstat


def sha256_file(path: Path) -> str:
    h = hashlib.sha256()
    with path.open("rb") as f:
        for chunk in iter(lambda: f.read(1024 * 1024), b""):
            h.update(chunk)
    return h.hexdigest()


def main() -> int:
    ap = argparse.ArgumentParser()
    ap.add_argument("--out-dir", required=True)
    ap.add_argument("--deterministic", action="store_true")
    ap.add_argument("--fit", action="store_true")
    ap.add_argument("--fit-repeat", type=int, default=3)
    ap.add_argument("--pk-nobs", default="8,32,128", help="Comma-separated obs counts for PK 1-cpt.")
    ap.add_argument("--pk-2cpt-nobs", default="32,64", help="Comma-separated obs counts for PK 2-cpt.")
    ap.add_argument("--nlme-nsub", default="4,16", help="Comma-separated subject counts for NLME.")
    ap.add_argument("--nlme-nobs", type=int, default=6, help="Obs per subject for NLME.")
    ap.add_argument("--foce-nsub", default="30,100", help="Comma-separated subject counts for FOCE.")
    ap.add_argument("--saem-nsub", default="30,100", help="Comma-separated subject counts for SAEM.")
    ap.add_argument("--pop-nobs", type=int, default=8, help="Obs per subject for FOCE/SAEM.")
    ap.add_argument("--seed", type=int, default=0)
    ap.add_argument("--smoke", action="store_true", help="Reduced set for CI speed.")
    args = ap.parse_args()

    out_dir = Path(args.out_dir).resolve()
    cases_dir = out_dir / "cases"
    cases_dir.mkdir(parents=True, exist_ok=True)

    pk_sizes = [int(x.strip()) for x in str(args.pk_nobs).split(",") if x.strip()]
    pk_2cpt_sizes = [int(x.strip()) for x in str(args.pk_2cpt_nobs).split(",") if x.strip()]
    nlme_sizes = [int(x.strip()) for x in str(args.nlme_nsub).split(",") if x.strip()]
    foce_sizes = [int(x.strip()) for x in str(args.foce_nsub).split(",") if x.strip()]
    saem_sizes = [int(x.strip()) for x in str(args.saem_nsub).split(",") if x.strip()]

    if args.smoke:
        pk_sizes = pk_sizes[:1]
        pk_2cpt_sizes = pk_2cpt_sizes[:1]
        nlme_sizes = nlme_sizes[:1]
        foce_sizes = foce_sizes[:1]
        saem_sizes = saem_sizes[:1]

    run_py = Path(__file__).resolve().parent / "run.py"

    suite_cases: list[dict] = []

    # 1-cpt oral PK
    for n in pk_sizes:
        suite_cases.append({
            "case_id": f"pk_1c_oral_nobs_{n}",
            "args": ["--model", "pk", "--n-obs", str(n)],
        })

    # 2-cpt IV PK
    for n in pk_2cpt_sizes:
        suite_cases.append({
            "case_id": f"pk_2cpt_iv_nobs_{n}",
            "args": ["--model", "pk_2cpt_iv", "--n-obs", str(n)],
        })

    # 2-cpt oral PK
    for n in pk_2cpt_sizes:
        suite_cases.append({
            "case_id": f"pk_2cpt_oral_nobs_{n}",
            "args": ["--model", "pk_2cpt_oral", "--n-obs", str(n)],
        })

    # NLME 1-cpt
    for n in nlme_sizes:
        suite_cases.append({
            "case_id": f"nlme_1c_oral_nsub_{n}",
            "args": ["--model", "nlme", "--n-subjects", str(n), "--n-obs-per-subject", str(int(args.nlme_nobs))],
        })

    # FOCE 1-cpt population
    for n in foce_sizes:
        suite_cases.append({
            "case_id": f"foce_1c_oral_nsub_{n}",
            "args": [
                "--model", "foce", "--n-subjects", str(n),
                "--n-obs-per-subject", str(int(args.pop_nobs)),
                "--error-model", "additive", "--sigma", "0.3",
            ],
        })

    # SAEM 1-cpt population
    for n in saem_sizes:
        suite_cases.append({
            "case_id": f"saem_1c_oral_nsub_{n}",
            "args": [
                "--model", "saem", "--n-subjects", str(n),
                "--n-obs-per-subject", str(int(args.pop_nobs)),
                "--error-model", "additive", "--sigma", "0.3",
            ],
        })

    index_cases = []
    for c in suite_cases:
        case_id = c["case_id"]
        out_path = cases_dir / f"{case_id}.json"
        cmd = [
            sys.executable, str(run_py),
            "--case", case_id,
            "--out", str(out_path),
            "--seed", str(int(args.seed)),
            *c["args"],
        ]
        if args.deterministic:
            cmd.append("--deterministic")
        if args.fit:
            cmd.extend(["--fit", "--fit-repeat", str(int(args.fit_repeat))])

        print(f"  Running: {case_id} ...", flush=True)
        subprocess.check_call(cmd)

        obj = json.loads(out_path.read_text())
        sha = sha256_file(out_path)
        model = obj.get("model", {})
        timing = obj.get("timing", {}).get("nll_time_s_per_call", {})
        fit = obj.get("fit") or {}
        foce = obj.get("foce") or {}
        saem = obj.get("saem") or {}

        entry: dict[str, Any] = {
            "case": case_id,
            "path": os.path.relpath(out_path, out_dir),
            "sha256": sha,
            "n_subjects": int(model.get("n_subjects", 0)),
            "n_obs": int(model.get("n_obs", 0)),
            "n_params": int(model.get("n_params", 0)),
        }

        # NLL timing (individual models)
        if timing:
            entry["nll_time_s_per_call_nextstat"] = float(timing.get("nextstat", 0.0))

        # MLE fit timing
        if isinstance(fit, dict) and fit.get("status") == "ok":
            entry["fit_status"] = "ok"
            entry["fit_time_s_nextstat"] = float((fit.get("time_s", {}) or {}).get("nextstat", 0.0))
        elif isinstance(fit, dict) and fit.get("status"):
            entry["fit_status"] = str(fit.get("status", ""))

        # FOCE results
        if isinstance(foce, dict) and foce.get("status") == "ok":
            entry["foce_status"] = "ok"
            entry["foce_time_s"] = float((foce.get("time_s", {}) or {}).get("nextstat", 0.0))
            rec = foce.get("recovery", {})
            max_rel = max((v.get("rel_err", 0.0) for v in rec.values()), default=0.0)
            entry["foce_max_rel_err"] = float(max_rel)
        elif isinstance(foce, dict) and foce.get("status"):
            entry["foce_status"] = str(foce.get("status", ""))

        # SAEM results
        if isinstance(saem, dict) and saem.get("status") == "ok":
            entry["saem_status"] = "ok"
            entry["saem_time_s"] = float((saem.get("time_s", {}) or {}).get("nextstat", 0.0))
            rec = saem.get("recovery", {})
            max_rel = max((v.get("rel_err", 0.0) for v in rec.values()), default=0.0)
            entry["saem_max_rel_err"] = float(max_rel)
        elif isinstance(saem, dict) and saem.get("status"):
            entry["saem_status"] = str(saem.get("status", ""))

        index_cases.append(entry)

    meta = {
        "python": sys.version.split()[0],
        "platform": platform.platform(),
        "nextstat_version": nextstat.__version__,
    }
    index = {
        "schema_version": "nextstat.pharma_benchmark_suite_result.v2",
        "suite": "pharma",
        "deterministic": bool(args.deterministic),
        "meta": meta,
        "cases": index_cases,
        "summary": {"n_cases": len(index_cases)},
    }
    (out_dir / "pharma_suite.json").write_text(json.dumps(index, indent=2, sort_keys=True) + "\n")
    return 0


if __name__ == "__main__":
    raise SystemExit(main())
