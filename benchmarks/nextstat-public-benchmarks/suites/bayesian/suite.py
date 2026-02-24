#!/usr/bin/env python3
"""Bayesian suite runner (seed).

Runs a small set of NUTS baselines intended to cover different geometry regimes:

1) Simple HistFactory (few params)
2) Logistic regression (GLM-style)
3) Hierarchical random intercepts (non-centered)

Writes:
- per-case JSON files (bayesian_benchmark_result_v1)
- a suite index JSON (bayesian_benchmark_suite_result_v1)
"""

from __future__ import annotations

import argparse
import hashlib
import json
import math
import os
import platform
import subprocess
import sys
from pathlib import Path

import nextstat


def sha256_file(path: Path) -> str:
    h = hashlib.sha256()
    with path.open("rb") as f:
        for chunk in iter(lambda: f.read(1024 * 1024), b""):
            h.update(chunk)
    return h.hexdigest()


def _maybe_float(x):
    try:
        v = float(x)
    except Exception:
        return None
    if not math.isfinite(v):
        return None
    return float(v)


def main() -> int:
    ap = argparse.ArgumentParser()
    ap.add_argument("--out-dir", required=True, help="Output directory.")
    ap.add_argument("--deterministic", action="store_true")
    ap.add_argument(
        "--backends",
        default="nextstat",
        help="Comma-separated list of backends: nextstat,nextstat_dense,cmdstanpy,pymc,numpyro (optional).",
    )
    ap.add_argument("--n-chains", type=int, default=4)
    ap.add_argument("--warmup", type=int, default=500)
    ap.add_argument("--samples", type=int, default=1000)
    ap.add_argument("--seed", type=int, default=0)
    ap.add_argument(
        "--dataset-seed",
        type=int,
        default=12345,
        help="Seed used for generated datasets (kept fixed across chain seeds).",
    )
    ap.add_argument("--max-treedepth", type=int, default=10)
    ap.add_argument("--target-accept", type=float, default=0.8)
    ap.add_argument("--init-jitter-rel", type=float, default=0.10)
    ap.add_argument(
        "--parity-warn-z",
        type=float,
        default=8.0,
        help="Warn if nextstat_dense vs nextstat max z-score exceeds this (posterior mean parity).",
    )
    ap.add_argument(
        "--parity-fail-z",
        type=float,
        default=12.0,
        help="Fail if nextstat_dense vs nextstat max z-score exceeds this (posterior mean parity).",
    )
    args = ap.parse_args()

    out_dir = Path(args.out_dir).resolve()
    cases_dir = out_dir / "cases"
    cases_dir.mkdir(parents=True, exist_ok=True)

    run_py = Path(__file__).resolve().parent / "run.py"

    cases = [
        {"case_id": "histfactory_simple_8p", "model": "histfactory_simple"},
        {"case_id": "glm_logistic_regression", "model": "glm_logistic"},
        {"case_id": "hier_random_intercept_non_centered", "model": "hier_random_intercept"},
        {"case_id": "eight_schools_non_centered", "model": "eight_schools"},
    ]

    index_cases = []
    n_ok = 0
    n_warn = 0
    n_failed = 0

    worst_case = "none"
    worst_score = float("inf")

    backends = [b.strip() for b in str(args.backends).split(",") if b.strip()]
    allowed = {"nextstat", "nextstat_dense", "cmdstanpy", "pymc", "numpyro"}
    for b in backends:
        if b not in allowed:
            raise SystemExit(f"unknown backend: {b} (allowed: {sorted(allowed)})")

    for c in cases:
        case_id = c["case_id"]
        for backend in backends:
            out_name = f"{case_id}.json" if backends == ["nextstat"] else f"{case_id}__{backend}.json"
            out_path = cases_dir / out_name
            cmd = [
                sys.executable,
                str(run_py),
                "--case",
                case_id,
                "--model",
                c["model"],
                "--backend",
                backend,
                "--out",
                str(out_path),
                "--n-chains",
                str(int(args.n_chains)),
                "--warmup",
                str(int(args.warmup)),
                "--samples",
                str(int(args.samples)),
                "--seed",
                str(int(args.seed)),
                "--dataset-seed",
                str(int(args.dataset_seed)),
                "--max-treedepth",
                str(int(args.max_treedepth)),
                "--target-accept",
                str(float(args.target_accept)),
                "--init-jitter-rel",
                str(float(args.init_jitter_rel)),
            ]
            if args.deterministic:
                cmd.append("--deterministic")
            p = subprocess.run(cmd)

            # Always include artifact if it exists (even if non-zero), but reflect status.
            try:
                obj = json.loads(out_path.read_text())
            except Exception:
                obj = {}

            status = str(obj.get("status") or "failed")
            if status == "ok":
                n_ok += 1
            elif status == "warn":
                n_warn += 1
            else:
                n_failed += 1

            wall = _maybe_float(obj.get("timing", {}).get("wall_time_s"))
            min_ess_bulk = _maybe_float(obj.get("diagnostics_summary", {}).get("min_ess_bulk"))
            min_ess_tail = _maybe_float(obj.get("diagnostics_summary", {}).get("min_ess_tail"))
            max_r_hat = _maybe_float(obj.get("diagnostics_summary", {}).get("max_r_hat"))
            min_ess_bulk_per_sec = _maybe_float(obj.get("timing", {}).get("ess_bulk_per_sec", {}).get("min"))

            # Worst-case heuristic: failed always worst; otherwise lowest ESS/sec when present.
            score = -1.0 if status == "failed" else float(min_ess_bulk_per_sec) if min_ess_bulk_per_sec is not None else float("inf")
            if score < worst_score:
                worst_score = score
                worst_case = f"{case_id}::{backend}"

            n_grad_evals = _maybe_float(obj.get("timing", {}).get("n_grad_evals"))
            ess_per_grad = _maybe_float(obj.get("timing", {}).get("ess_per_grad"))
            grad_per_sec = _maybe_float(obj.get("timing", {}).get("grad_per_sec"))

            sha = sha256_file(out_path) if out_path.exists() else "0" * 64
            index_cases.append(
                {
                    "case": case_id,
                    "backend": backend,
                    "path": os.path.relpath(out_path, out_dir),
                    "sha256": sha,
                    "status": status,
                    "wall_time_s": wall,
                    "min_ess_bulk": min_ess_bulk,
                    "min_ess_tail": min_ess_tail,
                    "max_r_hat": max_r_hat,
                    "min_ess_bulk_per_sec": min_ess_bulk_per_sec,
                    "n_grad_evals": int(n_grad_evals) if n_grad_evals is not None else None,
                    "ess_per_grad": ess_per_grad,
                    "grad_per_sec": grad_per_sec,
                }
            )

            # If the runner itself failed hard, bubble up.
            if int(p.returncode) != 0 and status == "failed":
                # Continue collecting the rest for debugging, but preserve suite failure.
                pass

    # ------------------------------------------------------------------
    # NextStat-only parity: diagonal vs dense metric should match posterior.
    # We compute mean z-scores using MC SE estimated via ESS_bulk when available.
    # This check is only meaningful when both variants are present + ok.
    # ------------------------------------------------------------------

    def _safe_float(x):
        try:
            v = float(x)
        except Exception:
            return None
        return v if math.isfinite(v) else None

    def _load_json(path: Path) -> dict:
        try:
            return json.loads(path.read_text())
        except Exception:
            return {}

    def _params_map(obj: dict) -> dict[str, dict]:
        diag = obj.get("diagnostics") if isinstance(obj.get("diagnostics"), dict) else {}
        ps = diag.get("posterior_summary") if isinstance(diag.get("posterior_summary"), dict) else {}
        if ps.get("status") != "ok":
            return {}
        params = ps.get("params")
        if not isinstance(params, list):
            return {}
        out: dict[str, dict] = {}
        for r in params:
            if isinstance(r, dict) and isinstance(r.get("name"), str):
                out[r["name"]] = r
        return out

    def _mc_se(row: dict, fallback_n: int) -> float | None:
        sd = _safe_float(row.get("sd"))
        if sd is None:
            return None
        ess = _safe_float(row.get("ess_bulk"))
        n_eff = ess if ess and ess > 1.0 else float(fallback_n)
        if n_eff <= 1.0:
            return None
        return sd / math.sqrt(n_eff)

    # Lookup backend artifacts for each case.
    lookup: dict[tuple[str, str], Path] = {}
    for e in index_cases:
        case = str(e.get("case"))
        backend = str(e.get("backend"))
        p = out_dir / str(e.get("path"))
        lookup[(case, backend)] = p

    parity_rows: list[dict] = []
    n_parity_warn = 0
    n_parity_fail = 0

    for c in cases:
        case_id = c["case_id"]
        p_diag = lookup.get((case_id, "nextstat"))
        p_dense = lookup.get((case_id, "nextstat_dense"))
        if not p_diag or not p_dense or not p_diag.exists() or not p_dense.exists():
            continue

        jd = _load_json(p_diag)
        je = _load_json(p_dense)
        if jd.get("status") != "ok" or je.get("status") != "ok":
            continue

        pm = _params_map(jd)
        pn = _params_map(je)
        if not pm or not pn:
            parity_rows.append(
                {"case": case_id, "status": "warn", "reason": "missing_posterior_summary", "max_z": None, "worst": []}
            )
            n_parity_warn += 1
            continue

        fallback_n = int(args.n_chains) * int(args.samples)
        max_z = None
        worst: list[tuple[float, str]] = []
        for name in sorted(set(pm.keys()) & set(pn.keys())):
            md = _safe_float(pm[name].get("mean"))
            mn = _safe_float(pn[name].get("mean"))
            if md is None or mn is None:
                continue
            se_d = _mc_se(pm[name], fallback_n)
            se_n = _mc_se(pn[name], fallback_n)
            if se_d is None or se_n is None:
                continue
            denom = math.sqrt(se_d * se_d + se_n * se_n)
            if denom <= 0:
                continue
            z = abs(md - mn) / denom
            max_z = z if max_z is None else max(max_z, z)
            worst.append((z, name))

        worst.sort(reverse=True)
        worst_top = [{"param": n, "z": float(z)} for z, n in worst[:3]]

        status = "ok"
        if max_z is None:
            status = "warn"
            n_parity_warn += 1
        elif max_z >= float(args.parity_fail_z):
            status = "failed"
            n_parity_fail += 1
        elif max_z >= float(args.parity_warn_z):
            status = "warn"
            n_parity_warn += 1

        parity_rows.append(
            {
                "case": case_id,
                "status": status,
                "max_z": float(max_z) if max_z is not None else None,
                "worst": worst_top,
            }
        )

    meta = {
        "python": sys.version.split()[0],
        "platform": platform.platform(),
        "nextstat_version": nextstat.__version__,
    }

    index = {
        "schema_version": "nextstat.bayesian_benchmark_suite_result.v1",
        "suite": "bayesian",
        "deterministic": bool(args.deterministic),
        "meta": meta,
        "cases": index_cases,
        "summary": {
            "n_cases": len(index_cases),
            "n_ok": n_ok,
            "n_warn": n_warn,
            "n_failed": n_failed,
            "worst_case": worst_case,
            "n_parity_warn": n_parity_warn,
            "n_parity_fail": n_parity_fail,
        },
        "parity": {
            "method": "mean_zscore",
            "compare": "nextstat_dense vs nextstat",
            "warn_z": float(args.parity_warn_z),
            "fail_z": float(args.parity_fail_z),
            "rows": parity_rows,
        },
    }

    index_path = out_dir / "bayesian_suite.json"
    index_path.write_text(json.dumps(index, indent=2, sort_keys=True) + "\n")

    # Only enforce parity failures when we actually ran both variants at production-like settings.
    parity_enforced = bool("nextstat_dense" in backends) and int(args.n_chains) >= 4 and int(args.samples) >= 1000
    if parity_enforced:
        return 0 if (n_failed == 0 and n_parity_fail == 0) else 2
    return 0 if n_failed == 0 else 2


if __name__ == "__main__":
    raise SystemExit(main())
