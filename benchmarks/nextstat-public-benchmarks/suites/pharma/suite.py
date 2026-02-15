#!/usr/bin/env python3
"""Pharma suite runner (seed).

Runs multiple generated PK/NLME cases and writes:
- per-case JSON files (pharma_benchmark_result_v2)
- a suite index JSON (pharma_benchmark_suite_result_v2)
- optional external baseline JSON files under `out_dir/baselines`

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


def _write_baseline_failed(path: Path, *, baseline: str, case_id: str, reason: str) -> None:
    obj = {
        "schema_version": "nextstat.pharma_baseline_result.v1",
        "baseline": baseline,
        "case": case_id,
        "status": "failed",
        "reason": reason,
    }
    path.parent.mkdir(parents=True, exist_ok=True)
    path.write_text(json.dumps(obj, indent=2, sort_keys=True) + "\n")


def main() -> int:
    ap = argparse.ArgumentParser()
    ap.add_argument("--out-dir", required=True)
    ap.add_argument("--deterministic", action="store_true")
    ap.add_argument("--fit", action="store_true")
    ap.add_argument("--fit-repeat", type=int, default=3)
    ap.add_argument("--run-baselines", action="store_true", help="Run external R baselines (nlmixr2/torsten) on FOCE/SAEM cases.")
    ap.add_argument("--baselines", default="nlmixr2,torsten", help="Comma-separated baselines to run when --run-baselines is set.")
    ap.add_argument("--baseline-repeat", type=int, default=5, help="Timed repeat count for baseline fits.")
    ap.add_argument("--torsten-iter", type=int, default=1200, help="Optimizer iterations for torsten/cmdstan baseline.")
    ap.add_argument("--rscript", default="Rscript", help="Rscript executable for baseline runners.")
    ap.add_argument("--baseline-r-libs-user", default="", help="Optional R_LIBS_USER for baseline runs.")
    ap.add_argument("--baseline-cmdstan", default="", help="Optional CMDSTAN path for torsten baseline.")
    ap.add_argument("--baseline-continue-on-error", action="store_true", help="Keep suite running if a baseline command errors.")
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

    baseline_entries: list[dict[str, Any]] = []
    if args.run_baselines:
        baselines_dir = out_dir / "baselines"
        baselines_dir.mkdir(parents=True, exist_ok=True)
        requested = [x.strip().lower() for x in str(args.baselines).split(",") if x.strip()]
        allowed = {"nlmixr2", "torsten"}
        enabled = [b for b in requested if b in allowed]
        unknown = [b for b in requested if b not in allowed]
        if unknown:
            print(f"  [warn] ignoring unknown baselines: {','.join(unknown)}", flush=True)

        base_env = os.environ.copy()
        if str(args.baseline_r_libs_user).strip():
            base_env["R_LIBS_USER"] = str(args.baseline_r_libs_user).strip()
        if str(args.baseline_cmdstan).strip():
            base_env["CMDSTAN"] = str(args.baseline_cmdstan).strip()

        root = Path(__file__).resolve().parent
        nlmixr2_runner = root / "baselines" / "nlmixr2" / "run.R"
        torsten_runner = root / "baselines" / "torsten" / "run.R"

        for c in suite_cases:
            case_id = str(c["case_id"])
            if not (case_id.startswith("foce_") or case_id.startswith("saem_")):
                continue
            case_json = cases_dir / f"{case_id}.json"
            if not case_json.exists():
                continue

            for baseline in enabled:
                out_path = baselines_dir / f"{baseline}_{case_id}.json"
                if baseline == "nlmixr2":
                    method = "focei" if case_id.startswith("foce_") else "saem"
                    cmd = [
                        str(args.rscript), str(nlmixr2_runner),
                        "--in", str(case_json),
                        "--out", str(out_path),
                        "--method", method,
                        "--repeat", str(int(args.baseline_repeat)),
                    ]
                else:
                    cmd = [
                        str(args.rscript), str(torsten_runner),
                        "--in", str(case_json),
                        "--out", str(out_path),
                        "--repeat", str(int(args.baseline_repeat)),
                        "--iter", str(int(args.torsten_iter)),
                    ]

                print(f"  Baseline: {baseline} / {case_id} ...", flush=True)
                try:
                    subprocess.check_call(cmd, env=base_env)
                except Exception as e:
                    reason = f"runner_error:{type(e).__name__}:{e}"
                    _write_baseline_failed(out_path, baseline=baseline, case_id=case_id, reason=reason)
                    if not args.baseline_continue_on_error:
                        raise

                b_obj: dict[str, Any] = {"status": "unknown", "reason": ""}
                try:
                    b_obj = json.loads(out_path.read_text())
                except Exception as e:
                    b_obj = {"status": "failed", "reason": f"json_parse_error:{type(e).__name__}:{e}"}

                baseline_entries.append({
                    "baseline": baseline,
                    "case": case_id,
                    "path": os.path.relpath(out_path, out_dir),
                    "sha256": sha256_file(out_path) if out_path.exists() else "",
                    "status": str(b_obj.get("status", "unknown")),
                    "reason": str(b_obj.get("reason", "")),
                    "fit_time_s": float(((b_obj.get("timing") or {}).get("fit_time_s", 0.0) or 0.0))
                    if isinstance(b_obj, dict) else 0.0,
                })

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
    if baseline_entries:
        n_ok = sum(1 for x in baseline_entries if x["status"] == "ok")
        n_warn = sum(1 for x in baseline_entries if x["status"] == "skipped")
        n_failed = sum(1 for x in baseline_entries if x["status"] == "failed")
        index["baselines"] = baseline_entries
        index["summary"]["baselines"] = {
            "n_runs": len(baseline_entries),
            "n_ok": int(n_ok),
            "n_warn": int(n_warn),
            "n_failed": int(n_failed),
        }
    (out_dir / "pharma_suite.json").write_text(json.dumps(index, indent=2, sort_keys=True) + "\n")
    return 0


if __name__ == "__main__":
    raise SystemExit(main())
