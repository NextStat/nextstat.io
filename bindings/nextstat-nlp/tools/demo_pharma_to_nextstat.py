#!/usr/bin/env python3
"""End-to-end demo: internet text -> nextstat-nlp -> nextstat (MLE).

This is a *workflow smoke*, not a benchmark:
- Fetch small public snippets (OpenFDA + ClinicalTrials.gov)
- Extract (regimens + survival) via nextstat-nlp (heuristic/onnx/gliner2/mlx)
- Convert into nextstat-ready inputs and run a tiny fit

Notes:
- Real PK NLME needs observation data; we generate a minimal synthetic dataset
  to demonstrate that the extracted regimen dict shape is accepted by nextstat.
"""

from __future__ import annotations

import argparse
import json
import math
import random
from dataclasses import asdict
from pathlib import Path
from typing import Any, Dict, List, Optional, Sequence

import requests

import nextstat
from nextstat_nlp import extract_regimens, extract_survival_records


def _write_json(path: Path, obj: Any) -> None:
    path.parent.mkdir(parents=True, exist_ok=True)
    path.write_text(json.dumps(obj, indent=2, sort_keys=True) + "\n", encoding="utf-8")


def _load_sources_json(path: Path) -> List[Dict[str, Any]]:
    raw = json.loads(path.read_text(encoding="utf-8"))
    if not isinstance(raw, list):
        raise ValueError("sources.json must be a list[dict]")
    return [r for r in raw if isinstance(r, dict)]


def _fetch_ctgov_intervention_text(nct_id: str) -> str:
    url = f"https://clinicaltrials.gov/api/v2/studies/{nct_id}"
    r = requests.get(url, timeout=30)
    r.raise_for_status()
    d = r.json()
    ps = d.get("protocolSection") or {}
    ident = ps.get("identificationModule") or {}
    arms = ps.get("armsInterventionsModule") or {}
    interventions = arms.get("interventions") or []
    lines: List[str] = []
    if ident.get("briefTitle"):
        lines.append(str(ident["briefTitle"]).strip())
    for it in interventions:
        name = (it.get("name") or "").strip()
        desc = (it.get("description") or "").strip()
        if name and desc:
            lines.append(f"{name}: {desc}")
        elif desc:
            lines.append(desc)
    return "\n".join([ln for ln in lines if ln])


def _synthetic_survival_texts() -> List[str]:
    return [
        "Subject 001: progressed at day 84. Age 63. Dose 20 mg daily. ECOG 1.",
        "Subject 002: lost to follow-up at 12 weeks. Age 58. Dose 10 mg daily. ECOG 0.",
        "Subject 003: died after 6 months. Age 71. Dose 5 mg/kg weekly. Weight 82 kg.",
        "Subject 004: withdrew at 48 hours. Age 45. Dose 500 mg IV.",
    ]


def _first_oral_regimen(reg_table: Any) -> Optional[Dict[str, Any]]:
    # reg_table is a RegimenTable dataclass; keep it loose here.
    for r in getattr(reg_table, "records", []):
        if str(getattr(r, "route", "") or "").strip().lower() == "oral":
            amount = float(getattr(r, "dose", 0.0) or 0.0)
            if amount > 0:
                return {
                    "dose_mg": amount,
                    "freq": str(getattr(r, "frequency", "") or ""),
                    "units": str(getattr(r, "amount_units", "") or ""),
                    "record": r,
                }
    return None


def _demo_survival_fit(out_dir: Path, *, backend: str, providers: Optional[Sequence[str]], num_threads: int) -> None:
    texts = _synthetic_survival_texts()
    ds = extract_survival_records(
        texts,
        backend=backend,  # type: ignore[arg-type]
        providers=providers,
        num_threads=num_threads,
        document_id="demo_survival_v1",
    )
    time, event, X, features = ds.to_design_matrix()
    fit = nextstat.survival.cox_ph.fit(time, event, X, ties="efron", robust=True)
    _write_json(out_dir / "demo_survival_inputs.json", {"time": time, "event": event, "X": X, "features": features})
    _write_json(out_dir / "demo_survival_fit.json", fit.__dict__)


def _demo_pk_fit_with_regimen_shape(
    out_dir: Path,
    *,
    backend: str,
    providers: Optional[Sequence[str]],
    num_threads: int,
    seed: int,
    sources_json: Optional[Path],
) -> None:
    # Use a registry snippet that contains an explicit regimen with infusion,
    # then use the *oral* record (if present) for a 1cpt_oral synthetic fit.
    # (We keep the NLME demo minimal and deterministic.)
    if sources_json is not None:
        sources = _load_sources_json(sources_json)
        # Use the same regimen-focused snippets as run_internet_mocks.py.
        texts = [
            (s.get("regimen_text") or s.get("text"))
            for s in sources
            if isinstance(s.get("text"), str)
        ]
        texts.append("Theophylline extended-release 400 mg orally once daily.")
        _write_json(out_dir / "demo_pk_sources.json", {"sources_json": str(sources_json), "n_sources": len(sources)})
    else:
        texts = [
            _fetch_ctgov_intervention_text("NCT01275781"),
            "Theophylline extended-release 400 mg orally once daily.",
        ]
        _write_json(out_dir / "demo_pk_sources.json", {"ctgov": "NCT01275781", "note": "fetched live"})

    reg = extract_regimens(
        texts,
        backend=backend,  # type: ignore[arg-type]
        providers=providers,
        num_threads=num_threads,
        document_id="demo_regimens_v1",
    )
    oral = _first_oral_regimen(reg)
    if oral is None:
        _write_json(out_dir / "demo_pk_note.json", {"error": "no oral regimen found in extracted records"})
        return

    dose = float(oral["dose_mg"])
    rng = random.Random(seed)

    # Synthetic observation design
    times = [0.5, 1.0, 2.0, 4.0, 8.0, 12.0, 24.0]
    true_theta = [0.04, 0.45, 1.5]  # CL, V, Ka (toy values)
    sigma = 0.10  # proportional SD

    # Generate observations from the analytic 1cpt oral model.
    dummy_y = [1.0] * len(times)
    m = nextstat.OneCompartmentOralPkModel(times, dummy_y, dose=dose, bioavailability=1.0, sigma=sigma)
    y_true = m.predict(true_theta)
    y_obs: List[float] = []
    for v in y_true:
        eps = rng.gauss(0.0, sigma)
        y_obs.append(max(float(v) * (1.0 + eps), 1e-6))

    # Build minimal nextstat regimens list accepted by nlme_foce.
    regimens = [{
        "events": [
            {"time": 0.0, "amount": dose, "route": "oral", "duration": 0.0},
        ]
    }]

    res = nextstat._core.nlme_foce(  # type: ignore[attr-defined]
        times=times,
        y=y_obs,
        subject_idx=[0] * len(times),
        n_subjects=1,
        model="1cpt_oral",
        method="focei",
        doses=[dose],
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

    # Compact diagnostics
    est = [float(v) for v in (res.get("theta") or [])]
    rel_err = [
        (abs(est[i] - true_theta[i]) / max(abs(true_theta[i]), 1e-12)) if i < len(est) else None
        for i in range(len(true_theta))
    ]

    _write_json(out_dir / "demo_pk_regimens.json", {"records": [asdict(r) for r in reg.records]})
    _write_json(out_dir / "demo_pk_inputs.json", {"dose": dose, "times": times, "y_obs": y_obs, "true_theta": true_theta, "sigma": sigma, "regimens": regimens})
    _write_json(out_dir / "demo_pk_fit.json", {"theta_est": est, "rel_err": rel_err, "converged": bool(res.get("converged")), "n_iter": int(res.get("n_iter") or 0)})


def main(argv: Optional[Sequence[str]] = None) -> int:
    ap = argparse.ArgumentParser(description="Demo: nextstat-nlp -> nextstat")
    ap.add_argument("--out-dir", type=Path, default=Path("tmp/nlp_demo_nextstat"))
    ap.add_argument("--backend", choices=["heuristic", "onnx", "gliner2", "mlx"], default="onnx")
    ap.add_argument("--num-threads", type=int, default=4)
    ap.add_argument("--providers", nargs="*", default=None)
    ap.add_argument("--seed", type=int, default=42)
    ap.add_argument("--sources-json", type=Path, default=None, help="Optional sources.json from run_internet_mocks.py for reproducible reruns.")
    args = ap.parse_args(list(argv) if argv is not None else None)

    out_dir: Path = args.out_dir
    backend: str = args.backend
    providers = args.providers if args.providers else None

    _demo_survival_fit(out_dir, backend=backend, providers=providers, num_threads=args.num_threads)
    _demo_pk_fit_with_regimen_shape(
        out_dir,
        backend=backend,
        providers=providers,
        num_threads=args.num_threads,
        seed=args.seed,
        sources_json=args.sources_json,
    )

    print(f"Wrote demo artifacts to: {out_dir}")
    return 0


if __name__ == "__main__":
    raise SystemExit(main())
