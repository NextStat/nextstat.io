#!/usr/bin/env python3
"""Reproducible workflow runner: internet sources -> nextstat-nlp -> nextstat.

This is the Apex2 verification harness for `nextstat-nlp`:
- Fetch internet snippets once (OpenFDA + ClinicalTrials.gov) into sources.json
- Re-run extraction `N` times offline to verify determinism (bit-exact JSON)
- Run a tiny nextstat fit (CoxPH + 1-subject NLME synthetic) from extracted structures

Outputs are designed to be copied as artifacts (not committed).
"""

from __future__ import annotations

import argparse
import hashlib
import json
import random
import subprocess
import sys
import time
from dataclasses import asdict
from pathlib import Path
from typing import Any, Dict, List, Optional, Sequence, Tuple

import nextstat
from nextstat_nlp import extract_regimens, extract_survival_records
from nextstat_nlp.backends import get_backend
from nextstat_nlp.priors import extract_prior_candidates
from nextstat_nlp.regimens import to_nextstat_regimens


def _sha256_file(path: Path) -> str:
    h = hashlib.sha256()
    with path.open("rb") as f:
        for chunk in iter(lambda: f.read(1 << 20), b""):
            h.update(chunk)
    return h.hexdigest()


def _run(cmd: Sequence[str], *, cwd: Optional[Path] = None) -> None:
    p = subprocess.run(list(cmd), cwd=str(cwd) if cwd else None)
    if p.returncode != 0:
        raise SystemExit(p.returncode)


def _load_json(path: Path) -> Any:
    return json.loads(path.read_text(encoding="utf-8"))


def _write_json(path: Path, obj: Any) -> None:
    path.parent.mkdir(parents=True, exist_ok=True)
    path.write_text(json.dumps(obj, indent=2, sort_keys=True) + "\n", encoding="utf-8")


def _collect_hashes(out_dir: Path) -> Dict[str, str]:
    keys = [
        "summary.json",
        "regimens.json",
        "survival.json",
        "priors.json",
        "nextstat_regimens.json",
        "nextstat_survival_design.json",
    ]
    out: Dict[str, str] = {}
    for k in keys:
        p = out_dir / k
        if p.exists():
            out[k] = _sha256_file(p)
    return out


def _synthetic_survival_mocks() -> List[str]:
    return [
        "Subject 001: progressed at day 84. Age 63. Dose 20 mg daily. ECOG 1. Stage IV.",
        "Subject 002: lost to follow-up at 12 weeks. Age 58. Dose 10 mg daily. ECOG 0.",
        "Subject 003: died after 6 months. Age 71. Dose 5 mg/kg weekly. Weight 82 kg.",
        "Subject 004: withdrew at 48 hours. Age 45. Dose 500 mg IV. Stage II.",
    ]


def _synthetic_prior_mocks() -> List[str]:
    return [
        "Clearance (CL) used a lognormal prior, mean 3.5 SD 1.2 L/h. Constraint: >0. Source: Smith 2020.",
        "V1 followed a normal distribution with mean 45 SD 12 L. Source: protocol section 2.3.",
        "ka prior: lognormal mean 1.5 SD 0.5 1/h. Constraint >0.",
    ]


def _extract_regimen_texts(sources: List[Dict[str, Any]]) -> List[str]:
    out: List[str] = []
    for s in sources:
        if not isinstance(s, dict):
            continue
        t = s.get("regimen_text") or s.get("text")
        if isinstance(t, str) and t.strip():
            out.append(t.strip())
    return out


def _summarize_regimens(records: Sequence[Dict[str, Any]]) -> Dict[str, Any]:
    n = len(records)
    n_route = sum(1 for r in records if (r.get("route") or "").strip())
    n_freq = sum(1 for r in records if (r.get("frequency") or "").strip())
    n_units = sum(1 for r in records if (r.get("amount_units") or "").strip())
    return {"n_records": n, "with_route": n_route, "with_frequency": n_freq, "with_units": n_units}


def _summarize_survival(records: Sequence[Dict[str, Any]]) -> Dict[str, Any]:
    n = len(records)
    n_cov = sum(1 for r in records if (r.get("covariates") or {}))
    cov_keys: Dict[str, int] = {}
    for r in records:
        cov = r.get("covariates") or {}
        for k in cov:
            cov_keys[k] = cov_keys.get(k, 0) + 1
    return {"n_records": n, "with_covariates": n_cov, "covariate_counts": dict(sorted(cov_keys.items()))}


def _summarize_priors(cands: Sequence[Dict[str, Any]]) -> Dict[str, Any]:
    n = len(cands)
    by_param: Dict[str, int] = {}
    for c in cands:
        p = str(c.get("param_name") or "")
        by_param[p] = by_param.get(p, 0) + 1
    return {"n_candidates": n, "by_param": dict(sorted(by_param.items()))}


def _first_oral_dose_mg(reg_records: Sequence[Dict[str, Any]]) -> Optional[float]:
    for r in reg_records:
        if str(r.get("route") or "").strip().lower() == "oral":
            try:
                v = float(r.get("dose") or 0.0)
            except Exception:
                continue
            if v > 0:
                return v
    return None


def _run_nextstat_demo(
    out_dir: Path,
    *,
    backend: str,
    backend_obj: Any,
    seed: int,
) -> Dict[str, Any]:
    timings: Dict[str, float] = {}

    # CoxPH on synthetic survival texts (this is the correct input contract)
    surv_texts = _synthetic_survival_mocks()
    t0 = time.perf_counter()
    surv = extract_survival_records(surv_texts, backend=backend, backend_obj=backend_obj, document_id="synthetic_survival_v1")
    time_arr, event_arr, X, features = surv.to_design_matrix()
    fit = nextstat.survival.cox_ph.fit(time_arr, event_arr, X, ties="efron", robust=True)
    timings["coxph_total_s"] = time.perf_counter() - t0
    _write_json(out_dir / "demo_survival_inputs.json", {"time": time_arr, "event": event_arr, "X": X, "features": features})
    _write_json(out_dir / "demo_survival_fit.json", fit.__dict__)

    # NLME: demonstrate that extracted regimen shape is accepted by nextstat.
    # We generate synthetic observations; this is a workflow smoke, not a scientific fit.
    reg_texts = [
        "Theophylline extended-release 400 mg orally once daily.",
    ]
    t1 = time.perf_counter()
    reg = extract_regimens(reg_texts, backend=backend, backend_obj=backend_obj, document_id="demo_regimens_v1")
    reg_j = asdict(reg)
    dose = _first_oral_dose_mg(reg_j.get("records") or [])
    if dose is None:
        _write_json(out_dir / "demo_pk_note.json", {"error": "no oral regimen found"})
        timings["nlme_total_s"] = time.perf_counter() - t1
        return {"timings": timings, "nlme": {"ran": False}}

    rng = random.Random(seed)
    times = [0.5, 1.0, 2.0, 4.0, 8.0, 12.0, 24.0]
    true_theta = [0.04, 0.45, 1.5]  # CL, V, Ka (toy values)
    sigma = 0.10  # proportional SD

    dummy_y = [1.0] * len(times)
    m = nextstat.OneCompartmentOralPkModel(times, dummy_y, dose=float(dose), bioavailability=1.0, sigma=sigma)
    y_true = m.predict(true_theta)
    y_obs: List[float] = []
    for v in y_true:
        eps = rng.gauss(0.0, sigma)
        y_obs.append(max(float(v) * (1.0 + eps), 1e-6))

    regimens = [{"events": [{"time": 0.0, "amount": float(dose), "route": "oral", "duration": 0.0}]}]
    res = nextstat._core.nlme_foce(  # type: ignore[attr-defined]
        times=times,
        y=y_obs,
        subject_idx=[0] * len(times),
        n_subjects=1,
        model="1cpt_oral",
        method="focei",
        doses=[float(dose)],
        regimens=regimens,
        error_model="proportional",
        sigma=sigma,
        estimate_sigma=False,
        theta_init=[0.05, 0.5, 1.2],
        omega_init=[0.05, 0.05, 0.05],
        max_outer_iter=50,
        max_inner_iter=20,
        tol=1e-4,
        rel_tol=1e-8,
        interaction=True,
    )

    est = [float(v) for v in (res.get("theta") or [])]
    rel_err = [
        (abs(est[i] - true_theta[i]) / max(abs(true_theta[i]), 1e-12)) if i < len(est) else None
        for i in range(len(true_theta))
    ]

    timings["nlme_total_s"] = time.perf_counter() - t1
    _write_json(out_dir / "demo_pk_inputs.json", {"dose": float(dose), "times": times, "y_obs": y_obs, "true_theta": true_theta, "sigma": sigma, "regimens": regimens})
    _write_json(out_dir / "demo_pk_fit.json", {"theta_est": est, "rel_err": rel_err, "converged": bool(res.get("converged")), "n_iter": int(res.get("n_iter") or 0)})
    return {"timings": timings, "nlme": {"ran": True, "converged": bool(res.get("converged"))}}


def main(argv: Optional[Sequence[str]] = None) -> int:
    ap = argparse.ArgumentParser(description="Run nextstat-nlp workflow matrix with reproducibility checks")
    ap.add_argument("--out-dir", type=Path, default=Path("tmp/nlp_workflow_matrix"))
    ap.add_argument("--python", type=str, default=sys.executable, help="Python executable for the initial online fetch step.")
    ap.add_argument("--sources-json", type=Path, default=None, help="Optional pre-fetched sources.json; if set, skips network fetch.")
    ap.add_argument("--n-repeats", type=int, default=3)
    ap.add_argument("--num-threads", type=int, default=4)
    ap.add_argument("--providers", nargs="*", default=None)
    ap.add_argument("--backends", nargs="*", default=["heuristic", "onnx"])
    ap.add_argument("--seed", type=int, default=42)
    args = ap.parse_args(list(argv) if argv is not None else None)

    root = Path(__file__).resolve().parents[1]
    tools_dir = root / "tools"
    out_root: Path = args.out_dir
    out_root.mkdir(parents=True, exist_ok=True)

    # 1) Acquire sources.json once (online), unless provided.
    if args.sources_json is not None:
        sources_json = args.sources_json
        if not sources_json.exists():
            raise SystemExit(f"--sources-json not found: {sources_json}")
    else:
        sources_run_dir = out_root / "sources_fetch"
        _run([args.python, str(tools_dir / "run_internet_mocks.py"), "--backend", "heuristic", "--out-dir", str(sources_run_dir)])
        sources_json = sources_run_dir / "sources.json"
        if not sources_json.exists():
            raise SystemExit(f"missing sources.json at {sources_json}")

    sources = _load_json(sources_json)
    if not isinstance(sources, list):
        raise SystemExit("sources.json must be a list[dict]")
    sources = [s for s in sources if isinstance(s, dict)]

    matrix: Dict[str, Any] = {
        "sources_json": str(sources_json),
        "runs": [],
    }
    providers: List[str] = list(args.providers) if args.providers else []

    # 2) Repeat offline runs for each backend and verify bit-exact outputs.
    for backend in args.backends:
        backend_runs: List[Dict[str, Any]] = []
        prev_hashes: Optional[Dict[str, str]] = None
        all_match = True

        # Reuse the backend instance in-process for all repeats.
        be = get_backend(
            backend,  # type: ignore[arg-type]
            providers=providers if providers else None,
            num_threads=args.num_threads,
        )

        for i in range(args.n_repeats):
            run_dir = out_root / f"{backend}/repeat_{i}"
            mocks_dir = run_dir / "internet_mocks"
            demo_dir = run_dir / "demo_nextstat"
            mocks_dir.mkdir(parents=True, exist_ok=True)
            demo_dir.mkdir(parents=True, exist_ok=True)

            reg_texts = _extract_regimen_texts(sources)
            surv_texts = _synthetic_survival_mocks()
            prior_texts = _synthetic_prior_mocks()

            # --- Extraction (offline; deterministic) ---
            t0 = time.perf_counter()
            reg = extract_regimens(
                reg_texts,
                backend=backend,  # type: ignore[arg-type]
                backend_obj=be,
                providers=providers if providers else None,
                num_threads=args.num_threads,
                document_id="internet_bundle_v1",
            )
            t_reg = time.perf_counter()
            surv = extract_survival_records(
                surv_texts,
                backend=backend,  # type: ignore[arg-type]
                backend_obj=be,
                providers=providers if providers else None,
                num_threads=args.num_threads,
                document_id="synthetic_survival_v1",
            )
            t_surv = time.perf_counter()
            pri = extract_prior_candidates(
                prior_texts,
                backend=backend,  # type: ignore[arg-type]
                backend_obj=be,
                providers=providers if providers else None,
                num_threads=args.num_threads,
                document_id="synthetic_priors_v1",
            )
            t_pri = time.perf_counter()

            reg_j = asdict(reg)
            surv_j = asdict(surv)
            pri_j = asdict(pri)
            nextstat_regs = to_nextstat_regimens(reg.records, expand_frequency=False)
            time_arr, event_arr, X, features = surv.to_design_matrix()

            summary = {
                "backend": backend,
                "backend_env": {"regimens": reg.backend_env, "survival": surv.backend_env, "priors": pri.backend_env},
                "counts": {
                    "n_sources": len(sources),
                    "n_regimen_texts": len(reg_texts),
                    "n_survival_texts": len(surv_texts),
                    "n_prior_texts": len(prior_texts),
                },
                "regimens": _summarize_regimens(reg_j.get("records") or []),
                "survival": _summarize_survival(surv_j.get("records") or []),
                "priors": _summarize_priors(pri_j.get("candidates") or []),
            }
            timings = {
                "extract_regimens": t_reg - t0,
                "extract_survival": t_surv - t_reg,
                "extract_priors": t_pri - t_surv,
                "extract_total": t_pri - t0,
            }

            _write_json(mocks_dir / "sources.json", sources)
            _write_json(mocks_dir / "regimens.json", reg_j)
            _write_json(mocks_dir / "survival.json", surv_j)
            _write_json(mocks_dir / "priors.json", pri_j)
            _write_json(mocks_dir / "nextstat_regimens.json", nextstat_regs)
            _write_json(mocks_dir / "nextstat_survival_design.json", {"time": time_arr, "event": event_arr, "X": X, "features": features})
            _write_json(mocks_dir / "summary.json", summary)
            _write_json(mocks_dir / "timings.json", timings)

            # --- NextStat smoke demo (offline; uses the same backend instance) ---
            demo = _run_nextstat_demo(demo_dir, backend=backend, backend_obj=be, seed=args.seed)

            h = _collect_hashes(mocks_dir)
            if prev_hashes is not None and h != prev_hashes:
                all_match = False
            prev_hashes = h

            backend_runs.append(
                {
                    "repeat": i,
                    "mocks_dir": str(mocks_dir),
                    "demo_dir": str(demo_dir),
                    "mocks_hashes": h,
                    "mocks_summary": summary,
                    "mocks_timings_s": timings,
                    "nextstat_demo": demo,
                }
            )

        matrix["runs"].append(
            {
                "backend": backend,
                "n_repeats": args.n_repeats,
                "bit_exact_mocks": bool(all_match),
                "runs": backend_runs,
            }
        )

    _write_json(out_root / "matrix_summary.json", matrix)
    print(f"Wrote workflow matrix artifacts to: {out_root}")
    return 0


if __name__ == "__main__":
    raise SystemExit(main())
