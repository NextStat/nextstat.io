#!/usr/bin/env python3
"""Run nextstat-nlp pipelines on public internet text snippets (mock workflow).

Goal: sanity-check that GLiNER2 helps extract nextstat-ready structured inputs
from real-world-ish pharma text sources (labels, trial registry).

This script intentionally avoids committing large third-party texts to the repo:
it fetches small snippets at runtime, records only hashes + extracted outputs.
"""

from __future__ import annotations

import argparse
import json
import re
import sys
from dataclasses import asdict
from pathlib import Path
from typing import Any, Dict, Iterable, List, Optional, Sequence, Tuple

import requests

from nextstat_nlp import extract_regimens, extract_survival_records
from nextstat_nlp.priors import extract_prior_candidates


def _write_json(path: Path, obj: Any) -> None:
    path.parent.mkdir(parents=True, exist_ok=True)
    path.write_text(json.dumps(obj, indent=2, sort_keys=True) + "\n", encoding="utf-8")


def _fetch_openfda_dosage(generic_name: str) -> Tuple[str, Dict[str, Any]]:
    url = "https://api.fda.gov/drug/label.json"
    params = {"search": f"openfda.generic_name:{generic_name}", "limit": 1}
    r = requests.get(url, params=params, timeout=30)
    r.raise_for_status()
    d = r.json()
    res = d["results"][0]
    dosage = (res.get("dosage_and_administration") or [""])[0]
    meta = {
        "source": "openfda.drug.label",
        "query": params["search"],
        "effective_time": res.get("effective_time"),
        "set_id": (res.get("openfda") or {}).get("set_id", [None])[0],
        "brand_name": (res.get("openfda") or {}).get("brand_name", [None])[0],
        "generic_name": (res.get("openfda") or {}).get("generic_name", [None])[0],
    }
    return dosage, meta


def _fetch_ctgov_interventions(nct_id: str) -> Tuple[str, Dict[str, Any]]:
    url = f"https://clinicaltrials.gov/api/v2/studies/{nct_id}"
    r = requests.get(url, timeout=30)
    r.raise_for_status()
    d = r.json()
    ps = d.get("protocolSection") or {}
    ident = ps.get("identificationModule") or {}
    desc = ps.get("descriptionModule") or {}
    arms = ps.get("armsInterventionsModule") or {}
    interventions = arms.get("interventions") or []
    lines: List[str] = []
    if ident.get("briefTitle"):
        lines.append(str(ident["briefTitle"]).strip())
    if desc.get("briefSummary"):
        lines.append(str(desc["briefSummary"]).strip())
    for it in interventions:
        name = (it.get("name") or "").strip()
        it_desc = (it.get("description") or "").strip()
        if name and it_desc:
            lines.append(f"{name}: {it_desc}")
        elif name:
            lines.append(name)
        elif it_desc:
            lines.append(it_desc)
    text = "\n".join([ln for ln in lines if ln])
    meta = {
        "source": "clinicaltrials.gov.api.v2",
        "nct_id": nct_id,
        "brief_title": ident.get("briefTitle"),
        "org": (ident.get("organization") or {}).get("fullName"),
    }
    return text, meta


def _synthetic_survival_mocks() -> List[str]:
    # One-record-per-subject style, compatible with SurvivalDataset contract.
    return [
        "Subject 001: progressed at day 84. Age 63. Dose 20 mg daily. ECOG 1. Stage IV.",
        "Subject 002: lost to follow-up at 12 weeks. Age 58. Dose 10 mg daily. ECOG 0.",
        "Subject 003: died after 6 months. Age 71. Dose 5 mg/kg weekly. Weight 82 kg.",
        "Subject 004: withdrew at 48 hours. Age 45. Dose 500 mg IV. Stage II.",
    ]


def _synthetic_prior_mocks() -> List[str]:
    # We keep these synthetic; real papers rarely put priors in plain English.
    return [
        "Clearance (CL) used a lognormal prior, mean 3.5 SD 1.2 L/h. Constraint: >0. Source: Smith 2020.",
        "V1 followed a normal distribution with mean 45 SD 12 L. Source: protocol section 2.3.",
        "ka prior: lognormal mean 1.5 SD 0.5 1/h. Constraint >0.",
    ]


def _normalize_snippet(text: str, *, max_chars: int = 5000) -> str:
    # Remove excessive whitespace but preserve enough context for NER.
    text = re.sub(r"[ \t]+", " ", text)
    text = re.sub(r"\n{3,}", "\n\n", text).strip()
    if len(text) > max_chars:
        return text[:max_chars] + "\n\n[TRUNCATED]\n"
    return text


def _extract_regimen_snippet(text: str, *, max_sentences: int = 4) -> str:
    """Pull a short, regimen-focused snippet from a larger block.

    OpenFDA labels can contain many section numbers and unrelated numeric values
    (INR targets, table values, references). Feeding the full block causes both
    heuristic and ML NER to over-generate dose candidates.
    """
    t = _normalize_snippet(text, max_chars=20_000)
    # Crude sentence split: good enough for the short blocks we fetch here.
    parts = re.split(r"(?<=[.!?])\s+|\n+", t)
    keep: List[str] = []
    for s in parts:
        s2 = s.strip()
        if not s2:
            continue
        s_l = s2.lower()
        has_unit = bool(re.search(r"\b(mg/kg|mg|mcg|ug|g|iu)\b", s_l))
        has_amount = bool(re.search(r"\b\d+(\.\d+)?\b", s_l))
        has_route_or_freq = any(
            k in s_l
            for k in (
                "intraven", " iv", "oral", "po", "infusion", "once", "daily", "weekly",
                "bid", "tid", "qid", "q12", "q8", "every", "hours", "hour", "over ",
            )
        )
        if has_unit and has_amount and has_route_or_freq:
            keep.append(s2)
        if len(keep) >= max_sentences:
            break
    if keep:
        return " ".join(keep)
    # Fallback: first ~600 chars gives the model some context but limits noise.
    return t[:600]


def _as_jsonable(obj: Any) -> Any:
    # nextstat-nlp outputs are frozen dataclasses; convert recursively.
    if hasattr(obj, "__dataclass_fields__"):
        return {k: _as_jsonable(v) for k, v in asdict(obj).items()}
    if isinstance(obj, dict):
        return {str(k): _as_jsonable(v) for k, v in obj.items()}
    if isinstance(obj, (list, tuple)):
        return [_as_jsonable(v) for v in obj]
    return obj


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


def _to_nextstat_regimens(regimens_json: Dict[str, Any]) -> List[Dict[str, Any]]:
    """Convert RegimenTable JSON into nextstat-style `regimens` dicts.

    This is intentionally a minimal mapping:
    - Frequency is preserved as metadata but not expanded into repeated events.
    - If start_time is missing, we assume time=0.
    - If route is missing, we keep it as "oral" (common default).
    """
    out: List[Dict[str, Any]] = []
    for r in regimens_json.get("records") or []:
        t0 = float(r.get("start_time") or 0.0)
        route = (r.get("route") or "oral").strip() or "oral"
        duration = float(r.get("duration") or 0.0)
        amount = float(r.get("dose") or 0.0)
        if amount <= 0:
            continue
        out.append(
            {
                "subject_id": r.get("subject_id"),
                "events": [
                    {
                        "time": t0,
                        "amount": amount,
                        "route": route,
                        "duration": duration,
                        # Leave bioavailability unset here; nextstat can infer defaults per route.
                    }
                ],
                "meta": {
                    "frequency": r.get("frequency"),
                    "amount_units": r.get("amount_units"),
                    "document_id": r.get("document_id"),
                },
            }
        )
    return out


def main(argv: Optional[Sequence[str]] = None) -> int:
    ap = argparse.ArgumentParser(description="Run nextstat-nlp on public internet snippets")
    ap.add_argument("--out-dir", type=Path, default=Path("tmp/nlp_internet_mocks"))
    ap.add_argument("--backend", choices=["onnx", "gliner2", "heuristic", "mlx"], default="onnx")
    ap.add_argument("--num-threads", type=int, default=4, help="ONNX Runtime threads (backend=onnx)")
    ap.add_argument("--providers", nargs="*", default=None, help="ONNX Runtime providers (backend=onnx)")
    ap.add_argument(
        "--sources-json",
        type=Path,
        default=None,
        help="Optional pre-fetched sources.json (list[dict]) to avoid network calls and make reruns reproducible.",
    )
    ap.add_argument("--openfda", nargs="*", default=["WARFARIN", "PHENOBARBITAL", "THEOPHYLLINE"])
    ap.add_argument("--ctgov", nargs="*", default=["NCT01275781", "NCT00964353"])
    args = ap.parse_args(list(argv) if argv is not None else None)

    out_dir: Path = args.out_dir
    backend: str = args.backend

    sources: List[Dict[str, Any]] = []

    if args.sources_json is not None:
        raw = json.loads(args.sources_json.read_text(encoding="utf-8"))
        if not isinstance(raw, list):
            raise SystemExit("--sources-json must be a list of dicts (sources.json)")
        # Trust file contents; keep this script a thin runner.
        sources = [s for s in raw if isinstance(s, dict)]
    else:
        # --- Internet snippets: OpenFDA labels ---
        for g in args.openfda:
            try:
                text, meta = _fetch_openfda_dosage(g)
            except Exception as e:
                sources.append({"kind": "openfda", "id": g, "error": str(e)})
                continue
            norm = _normalize_snippet(text)
            sources.append({
                "kind": "openfda",
                "id": g,
                "text": norm,
                "regimen_text": _extract_regimen_snippet(norm),
                "meta": meta,
            })

        # --- Internet snippets: ClinicalTrials.gov registry ---
        for nct in args.ctgov:
            try:
                text, meta = _fetch_ctgov_interventions(nct)
            except Exception as e:
                sources.append({"kind": "ctgov", "id": nct, "error": str(e)})
                continue
            norm = _normalize_snippet(text)
            sources.append({
                "kind": "ctgov",
                "id": nct,
                "text": norm,
                "regimen_text": _extract_regimen_snippet(norm),
                "meta": meta,
            })

    # --- Synthetic snippets for survival/priors (per-record format) ---
    survival_texts = _synthetic_survival_mocks()
    prior_texts = _synthetic_prior_mocks()

    providers = args.providers if args.providers else None

    # --- Run pipelines ---
    # Regimens: run on internet snippets (documents)
    reg_texts = [
        (s.get("regimen_text") or s.get("text"))
        for s in sources
        if isinstance(s.get("text"), str)
    ]
    reg = extract_regimens(
        reg_texts,
        backend=backend,  # type: ignore[arg-type]
        providers=providers,
        num_threads=args.num_threads,
        document_id="internet_bundle_v1",
    )
    reg_j = _as_jsonable(reg)

    # Survival: run on synthetic per-subject texts (this is the correct contract)
    surv = extract_survival_records(
        survival_texts,
        backend=backend,  # type: ignore[arg-type]
        providers=providers,
        num_threads=args.num_threads,
        document_id="synthetic_survival_v1",
    )
    surv_j = _as_jsonable(surv)

    pri = extract_prior_candidates(
        prior_texts,
        backend=backend,  # type: ignore[arg-type]
        providers=providers,
        num_threads=args.num_threads,
        document_id="synthetic_priors_v1",
    )
    pri_j = _as_jsonable(pri)

    summary = {
        "backend": backend,
        "backend_env": {
            "regimens": reg.backend_env,
            "survival": surv.backend_env,
            "priors": pri.backend_env,
        },
        "sources": [{"kind": s.get("kind"), "id": s.get("id"), "meta": s.get("meta"), "error": s.get("error")} for s in sources],
        "counts": {
            "n_sources": len(sources),
            "n_regimen_texts": len(reg_texts),
            "n_survival_texts": len(survival_texts),
            "n_prior_texts": len(prior_texts),
        },
        "regimens": _summarize_regimens(reg_j.get("records") or []),
        "survival": _summarize_survival(surv_j.get("records") or []),
        "priors": _summarize_priors(pri_j.get("candidates") or []),
    }

    # --- Write artifacts ---
    _write_json(out_dir / "sources.json", sources)
    _write_json(out_dir / "regimens.json", reg_j)
    _write_json(out_dir / "survival.json", surv_j)
    _write_json(out_dir / "priors.json", pri_j)
    _write_json(out_dir / "nextstat_regimens.json", _to_nextstat_regimens(reg_j))

    # Convenience: show survival arrays ready for modeling.
    time, event, X, features = surv.to_design_matrix()
    _write_json(out_dir / "nextstat_survival_design.json", {"time": time, "event": event, "X": X, "features": features})
    _write_json(out_dir / "summary.json", summary)

    # --- Print compact summary ---
    print(json.dumps(summary, indent=2, sort_keys=True))
    return 0


if __name__ == "__main__":
    raise SystemExit(main())
